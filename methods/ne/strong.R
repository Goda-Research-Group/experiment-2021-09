set.seed(1234)

library(mgcv)

args <- commandArgs(trailingOnly = T)
tmpdir = args[1]

f <- read.table(paste(tmpdir, "/inputF.txt", sep = ""))
y <- read.table(paste(tmpdir, "/inputY.txt", sep = ""))

fSize <- ncol(f)
ySize <- ncol(y)
N <- nrow(f)

g.mat <- matrix(NA, N, fSize)

for (i in 1:fSize){
    if (ySize == 1) {
        f.gam <- gam(f[, i] ~ te(y[, 1]))
        g.mat[, i] <- f.gam$fitted
    } else if (ySize == 2) {
        f.gam <- gam(f[, i] ~ te(y[, 1], y[, 2]))
        g.mat[, i] <- f.gam$fitted
    } else if (ySize == 3) {
        f.gam <- gam(f[, i] ~ te(y[, 1], y[, 2], y[, 3]))
        g.mat[, i] <- f.gam$fitted
    }
}

g <- as.data.frame(g.mat)

write.table(g, paste(tmpdir, "/outputF.txt", sep = ""), sep = " ", row.names = F, col.names = F)
