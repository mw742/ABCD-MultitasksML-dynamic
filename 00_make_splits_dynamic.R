#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(data.table) })

args <- commandArgs(trailingOnly=TRUE)
get_arg <- function(k, d=NULL) {
  hit <- grep(paste0("^--", k, "="), args, value=TRUE)
  if (length(hit)==0) return(d)
  sub(paste0("^--", k, "="), "", hit[1])
}

SURV_DIR <- get_arg("surv_dir", "/gpfs/group/ehlers/mwei/ABCD_2/ABCD/work_provided_assets/Env_surv")
OUT_DIR  <- get_arg("out_dir",  getwd())
SEED     <- as.integer(get_arg("seed", "42"))
FRAC_TRAIN <- as.numeric(get_arg("frac_train", "0.70"))
FRAC_VAL   <- as.numeric(get_arg("frac_val",   "0.15"))

if (FRAC_TRAIN <= 0 || FRAC_VAL <= 0 || FRAC_TRAIN + FRAC_VAL >= 1) {
  stop("Bad split fractions: need train>0, val>0, train+val<1")
}

set.seed(SEED)
msg <- function(...) message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste0(..., collapse=""))

f_any <- file.path(SURV_DIR, "env_surv_intervals_any.tsv")
if (!file.exists(f_any)) stop("Missing: ", f_any)

dt <- fread(f_any, showProgress=FALSE)
if (!("IID" %in% names(dt))) stop("env_surv_intervals_any.tsv missing IID")
dt <- unique(dt[, .(IID)])
setorder(dt, IID)

f_any_full <- fread(file.path(SURV_DIR, "env_surv_intervals_any.tsv"), showProgress=FALSE)
f_any_full[, event_interval := as.integer(as.numeric(event_interval))]
f_any_full[is.na(event_interval), event_interval := 0L]
ever <- f_any_full[, .(ever_event = as.integer(any(event_interval==1L))), by=IID]
dt <- merge(dt, ever, by="IID", all.x=TRUE)
dt[is.na(ever_event), ever_event := 0L]

split_one <- function(ids) {
  n <- length(ids)
  n_train <- floor(n * FRAC_TRAIN)
  n_val   <- floor(n * FRAC_VAL)
  perm <- sample(ids, size=n, replace=FALSE)
  list(
    train = perm[1:n_train],
    val   = perm[(n_train+1):(n_train+n_val)],
    test  = perm[(n_train+n_val+1):n]
  )
}

ids1 <- dt[ever_event==1L, IID]
ids0 <- dt[ever_event==0L, IID]
s1 <- split_one(ids1)
s0 <- split_one(ids0)

splits <- rbindlist(list(
  data.table(IID=c(s1$train, s0$train), split="train"),
  data.table(IID=c(s1$val,   s0$val),   split="val"),
  data.table(IID=c(s1$test,  s0$test),  split="test")
))

if (anyDuplicated(splits$IID)) stop("IID overlap across splits detected (bug).")

dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)
fwrite(splits, file.path(OUT_DIR, "splits.tsv"), sep="\t")

manifest <- data.table(
  param=c("SURV_DIR","SEED","FRAC_TRAIN","FRAC_VAL"),
  value=c(SURV_DIR, SEED, FRAC_TRAIN, FRAC_VAL)
)
fwrite(manifest, file.path(OUT_DIR, "splits_manifest.tsv"), sep="\t")

msg("Wrote splits.tsv")
print(splits[, .N, by=split][order(split)])

