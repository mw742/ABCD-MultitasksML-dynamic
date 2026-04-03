#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
})

ts_msg <- function(...) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste0(..., collapse=""))
}

args <- commandArgs(trailingOnly=TRUE)
get_arg <- function(key, default=NULL) {
  hit <- grep(paste0("^--", key, "="), args, value=TRUE)
  if (length(hit)==0) return(default)
  sub(paste0("^--", key, "="), "", hit[1])
}
`%||%` <- function(x,y) if (is.null(x) || length(x)==0 || is.na(x)) y else x

WORKDIR <- get_arg("workdir", getwd())
ENV_SURV_DIR <- get_arg("env_surv_dir",
  "/path/"
)

COV_FILE <- get_arg("cov_file", file.path(ENV_SURV_DIR, "cov_timevarying_filtered.tsv"))
CAND_FILE <- get_arg("cand_file", file.path(ENV_SURV_DIR, "candidate_exposures_shared.tsv"))

SPLITS_TSV <- get_arg("splits", file.path(WORKDIR, "splits.tsv"))
OUT_DIR <- get_arg("outdir", file.path(WORKDIR, "results"))

# Interval files produced by your survival pipeline (seen in Env_surv)
TRAITS <- c("alcohol","nicotine","cannabis","any")
INTERVAL_FILES <- setNames(file.path(ENV_SURV_DIR, paste0("env_surv_intervals_", TRAITS, ".tsv")), TRAITS)

CORE_COVARS <- c("sex","baseline_age","age","site", paste0("PC",1:20))

PRS_PATHS <- list(
  PRS_PAU      = "PRS_two_col.tsv",
  PRS_CanUD    = "PRS_two_col.tsv",
  PRS_nicotine = "PRS_two_col.tsv",
  PRS_anySUD   = "PRS_two_col.tsv"
)

dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

assert_no_dup_names <- function(dt, where="table") {
  dup <- names(dt)[duplicated(names(dt))]
  if (length(dup)>0) stop(where, " has duplicated column names: ", paste(unique(dup), collapse=", "))
}

safe_merge_by <- function(x, y, by, all.x=TRUE, all.y=FALSE) {
  assert_no_dup_names(x, "Left before merge")
  assert_no_dup_names(y, "Right before merge")
  if (!all(by %in% names(x))) stop("Left missing keys: ", paste(setdiff(by, names(x)), collapse=", "))
  if (!all(by %in% names(y))) stop("Right missing keys: ", paste(setdiff(by, names(y)), collapse=", "))
  dup <- intersect(setdiff(names(x), by), setdiff(names(y), by))
  if (length(dup)>0) stop("Refusing merge (would create .x/.y duplicates): ", paste(dup, collapse=", "))
  out <- merge(x, y, by=by, all.x=all.x, all.y=all.y)
  assert_no_dup_names(out, "Merged")
  out
}

safe_read_score_table <- function(path, score_name) {
  if (!file.exists(path)) {
    ts_msg("[warn] PRS missing, skipping: ", path)
    return(NULL)
  }

  # Read as-is, allow ragged rows
  dt <- fread(path, header=FALSE, fill=TRUE, data.table=TRUE)
  if (nrow(dt) == 0) return(NULL)

  # Drop completely empty columns
  empty_col <- vapply(dt, function(x) all(is.na(x) | trimws(as.character(x)) == ""), logical(1))
  dt <- dt[, which(!empty_col), with=FALSE]

  first_row <- tolower(trimws(as.character(unlist(dt[1,]))))
  if (any(first_row %in% c("fid","iid"))) {
    dt <- fread(path, header=TRUE, fill=TRUE, data.table=TRUE)
    setnames(dt, names(dt), make.names(names(dt), unique=TRUE))

    if (all(c("FID","IID") %in% names(dt))) {
      score_cols <- setdiff(names(dt), c("FID","IID"))
      if (length(score_cols) < 1) stop("PRS has no score column: ", path)

      sc <- score_cols[1]
      out <- dt[, .(FID=as.character(FID),
                    IID=as.character(IID),
                    score=suppressWarnings(as.numeric(get(sc))))]
      out <- out[!is.na(FID) & !is.na(IID)]
      out <- unique(out, by=c("FID","IID"))
      setnames(out, "score", score_name)
      return(out)
    }
  }

  if (ncol(dt) < 2) stop("Unrecognized PRS format (<2 cols): ", path)

  if (ncol(dt) == 2) {
    iid <- as.character(dt[[1]])
    sc  <- suppressWarnings(as.numeric(dt[[2]]))
    out <- data.table(FID=iid, IID=iid)
    out[, (score_name) := sc]
    out <- out[!is.na(FID) & !is.na(IID)]
    out <- unique(out, by=c("FID","IID"))
    return(out)
  }

  # ncol >= 3: take first 3 columns only
  fid <- as.character(dt[[1]])
  iid <- as.character(dt[[2]])
  sc  <- suppressWarnings(as.numeric(dt[[3]]))

  # If IID is missing, fall back to FID
  iid[is.na(iid) | iid==""] <- fid[is.na(iid) | iid==""]

  out <- data.table(FID=fid, IID=iid)
  out[, (score_name) := sc]
  out <- out[!is.na(FID) & !is.na(IID)]
  out <- unique(out, by=c("FID","IID"))
  return(out)
}


# --- 0) Splits ---
if (!file.exists(SPLITS_TSV)) stop("Missing splits.tsv: ", SPLITS_TSV)
splits <- fread(SPLITS_TSV)
if (!all(c("FID","IID","split") %in% names(splits))) stop("splits.tsv must have FID IID split")
splits[, FID:=as.character(FID)]
splits[, IID:=as.character(IID)]
splits[, split:=as.character(split)]
splits <- splits[, .(FID,IID,split)]
if (anyDuplicated(paste(splits$FID,splits$IID))>0) stop("Duplicate IDs in splits.tsv")
ts_msg("Loaded splits: ", nrow(splits), " subjects")

# --- 1) Read all interval files LONG, then make a MASTER grid, then WIDE labels ---
ts_msg("Reading interval files...")
int_list <- list()
for (tr in names(INTERVAL_FILES)) {
  f <- INTERVAL_FILES[[tr]]
  if (!file.exists(f)) {
    ts_msg("[warn] Missing interval file for ", tr, ": ", f, " (skip)")
    next
  }
  dt <- fread(f)
  # Expect at least: FID IID start stop event (names vary across pipelines)
  dt[, FID:=as.character(FID)]
  dt[, IID:=as.character(IID)]

  # Guess start/stop/event columns
  start_col <- c("tstart","start","start_time","t0","time_start","tstart_months")[c("tstart","start","start_time","t0","time_start","tstart_months") %in% names(dt)][1] %||% NA_character_
  stop_col  <- c("tstop","stop","stop_time","t1","time_stop","tstop_months")[c("tstop","stop","stop_time","t1","time_stop","tstop_months") %in% names(dt)][1] %||% NA_character_
  event_col <- c("event","status","d","fail","y")[c("event","status","d","fail","y") %in% names(dt)][1] %||% NA_character_

  if (is.na(start_col) || is.na(stop_col) || is.na(event_col)) {
    stop("Interval file missing required cols (need start/stop/event). File=", f,
         " found start=", start_col, " stop=", stop_col, " event=", event_col)
  }

  dt <- dt[, .(
    FID, IID,
    tstart = as.numeric(get(start_col)),
    tstop  = as.numeric(get(stop_col)),
    event  = as.integer(as.numeric(get(event_col)))
  )]

  dt[, trait := tr]
  int_list[[tr]] <- dt
}

if (length(int_list)==0) stop("No interval files loaded.")

int_long <- rbindlist(int_list, use.names=TRUE, fill=TRUE)
# Restrict to split subjects only (prevents leakage + reduces size)
int_long <- int_long[paste(FID,IID) %in% paste(splits$FID, splits$IID)]

# Make a stable interval key
int_long[, interval_key := paste(FID, IID, tstart, tstop, sep="|")]

# One row per (subject,interval,trait) → if duplicates exist, collapse (max event)
int_long <- int_long[, .(event=max(event, na.rm=TRUE)), by=.(FID,IID,tstart,tstop,interval_key,trait)]

# Pivot to wide labels: y_<trait>
ts_msg("Pivoting interval labels to wide multi-task columns...")
lab_wide <- dcast(int_long, FID + IID + tstart + tstop + interval_key ~ trait,
                  value.var="event", fill=0)

# Rename to y_*
for (tr in setdiff(names(lab_wide), c("FID","IID","tstart","tstop","interval_key"))) {
  setnames(lab_wide, tr, paste0("y_", tr))
}
label_cols <- grep("^y_", names(lab_wide), value=TRUE)
ts_msg("Dynamic label cols: ", paste(label_cols, collapse=", "))

# --- 2) Load covariates and join features at interval start (tstart) ---
if (!file.exists(COV_FILE)) stop("Missing covariates: ", COV_FILE)
ts_msg("Reading covariates: ", COV_FILE)
cov <- fread(COV_FILE)
if (!all(c("FID","IID","time_months") %in% names(cov))) stop("cov_timevarying_filtered.tsv must have FID IID time_months")
cov[, FID:=as.character(FID)]
cov[, IID:=as.character(IID)]
cov[, time_months := as.numeric(time_months)]
cov <- cov[paste(FID,IID) %in% paste(splits$FID,splits$IID)]

# Candidate exposures list
if (!file.exists(CAND_FILE)) stop("Missing candidate exposures: ", CAND_FILE)
cand <- fread(CAND_FILE)
if (!"term" %in% names(cand)) stop("candidate_exposures_shared.tsv must contain 'term'")
cand_terms <- unique(as.character(cand$term))

# Build a baseline feature template from cov columns (numeric only)
core_present <- intersect(CORE_COVARS, names(cov))
# Filter candidate terms to those present and numeric
exposures <- intersect(cand_terms, names(cov))
drop_like <- c("src_subject_id","eventname","interview_age","baseline_age","time_months")
exposures <- setdiff(exposures, drop_like)
is_num <- vapply(exposures, function(v) is.numeric(cov[[v]]) || is.integer(cov[[v]]), logical(1))
exposures <- exposures[is_num]

feat_cols <- unique(c(core_present, exposures))
ts_msg("Using candidate-only features: core(", length(core_present), ") + exposures(", length(exposures), ")")

# Create join key for cov at visit time
cov_key <- cov[, c("FID","IID","time_months", feat_cols), with=FALSE]
setnames(cov_key, "time_months", "tstart")
setkey(cov_key, FID, IID, tstart)

setkey(lab_wide, FID, IID, tstart)
joined <- cov_key[lab_wide, nomatch=0]  # join brings in lab_wide cols directly

need_cols <- c("tstop","interval_key", label_cols)
miss <- setdiff(need_cols, names(joined))
if (length(miss) > 0) stop("After join, missing columns: ", paste(miss, collapse=", "))


# --- 3) Merge PRS (static by subject) without V3.x/y duplicates ---
ts_msg("Merging PRS tables (if present)...")
prs_list <- list()
for (nm in names(PRS_PATHS)) {
  prs_dt <- safe_read_score_table(PRS_PATHS[[nm]], nm)
  if (!is.null(prs_dt)) {
    assert_no_dup_names(prs_dt, paste0("PRS ", nm))
    # refuse accidental overlap besides FID/IID
    bad <- setdiff(intersect(names(prs_dt), names(joined)), c("FID","IID"))
    if (length(bad) > 0) stop("PRS ", nm, " overlaps with joined cols: ", paste(bad, collapse=", "))
    prs_list[[nm]] <- prs_dt
    ts_msg("PRS loaded: ", nm, " rows=", nrow(prs_dt), " cols=", paste(names(prs_dt), collapse=","))
  }
}

if (length(prs_list) > 0) {
  prs_all <- Reduce(function(a,b) safe_merge_by(a,b, by=c("FID","IID"), all.x=TRUE, all.y=TRUE), prs_list)
  joined  <- safe_merge_by(joined, prs_all, by=c("FID","IID"), all.x=TRUE, all.y=FALSE)
}


# --- 4) Attach split (by subject) and write ---
joined <- safe_merge_by(joined, splits, by=c("FID","IID"), all.x=TRUE, all.y=FALSE)
if (any(is.na(joined$split))) stop("Some rows missing split after merge (splits.tsv inconsistent).")

# Final: ensure no duplicated names
assert_no_dup_names(joined, "Final dynamic dataset")

ts_msg("Final dynamic dataset rows: ", nrow(joined), " cols: ", ncol(joined))

out_full  <- file.path(OUT_DIR, "mtl_dynamic_full.tsv")
out_train <- file.path(OUT_DIR, "mtl_dynamic_train.tsv")
out_val   <- file.path(OUT_DIR, "mtl_dynamic_val.tsv")
out_test  <- file.path(OUT_DIR, "mtl_dynamic_test.tsv")

fwrite(joined, out_full,  sep="\t", quote=FALSE, na="NA")
fwrite(joined[split=="train"], out_train, sep="\t", quote=FALSE, na="NA")
fwrite(joined[split=="val"],   out_val,   sep="\t", quote=FALSE, na="NA")
fwrite(joined[split=="test"],  out_test,  sep="\t", quote=FALSE, na="NA")

ts_msg("DONE. Wrote dynamic datasets to: ", OUT_DIR)

