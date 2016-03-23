library(caret)
library(plyr)
library(doMC)
#Loading data
traind <- as.matrix(read.table('train.csv', sep = ',', header = TRUE, stringsAsFactors = FALSE))

pnum <- function(x) {   #x is each row in matrix; function plotting digits
        x <- matrix(x[-1], nrow = 28, ncol = 28, byrow = TRUE)
        x <- t(apply(x, 2, rev))
        image(x,col=grey.colors(255))
}

#Feature extraction from raw data

featt <- data.frame(   #Data table with extracted features
        label = as.factor(traind[,1]),
        size = rowSums(traind[,-1])
)

trainl <- split(traind[,-1], rep(1:nrow(traind)))
trainl <- lapply(trainl, function(x) matrix(x, nrow = 28, ncol = 28, byrow = T))

#Idea for second try would be changing color code from 0-255 gray palette to 0/1 black-white
dummy.col <- function(m){
        m[m[]>0] <- 1L
        return(m)
}
trainl <- lapply(trainl, dummy.col)
####
trimmer <- function(m) {
        rows <- apply(m,1,function(row) sum(row) > 0)
        cols <- apply(m,2,function(row) sum(row) > 0)
        return(m[rows,cols])
}

trainl <- lapply(trainl, trimmer)

featt$ratio <- as.numeric(lapply(trainl, function(x) nrow(x)/ncol(x)))

#Vertical symmetry of digit
symmetry.v <- function(m){   #my idea for vertical symmetry calculation
        nc <- floor(dim(m)[2]/2)
        fulnc <- dim(m)[2]
        a <- apply(m, 1, function(x) sum(x[1:nc])) 
        b <- apply(m, 1, function(x) sum(x[(nc+1):fulnc]))
        mean(abs(a-b))
}

featt$symmetry.vert <- as.numeric(lapply(trainl, symmetry.v))

#Horizontal symmetry of digit
symmetry.h <- function(m){
        nc <- floor(dim(m)[1]/2)
        fulnc <- dim(m)[1]
        a <- apply(m, 2, function(x) sum(x[1:nc])) 
        b <- apply(m, 2, function(x) sum(x[(nc+1):fulnc]))
        mean(abs(a-b))
}

featt$symmetry.horiz <- as.numeric(lapply(trainl, symmetry.h))

#Weight of pixels of whole digit
mean.point <- function(m){
        mean(m)/max(m)
}

featt$mean.point <- as.numeric(lapply(trainl, mean.point))

#Situation in left quarter (think of better representation)
quart.left <- function(m){
        nc <- floor(dim(m)[2]/2)
        nr <- floor(dim(m)[1]/2)
        mean(m[1:nr,1:nc])/(max(m[1:nr,1:nc]) + 1)
}
featt$quart.left <- as.numeric(lapply(trainl, quart.left))

#Situation in right quarter (think of better representation)
quart.right <- function(m){
        nc <- floor(dim(m)[2]/2)
        nr <- floor(dim(m)[1]/2)
        mean(m[1:nr,(nc+1):dim(m)[2]])/(max(m[1:nr,(nc+1):dim(m)[2]]) + 1)
}
featt$quart.right <- as.numeric(lapply(trainl, quart.right))

#Situation in bottom left quarter
quart.left.b <- function(m){
        nc <- floor(dim(m)[2]/2)
        nr <- floor(dim(m)[1]/2)
        mean(m[(nr+1):dim(m)[1],1:nc])/(max(m[(nr+1):dim(m)[1],1:nc]) + 1)
}
featt$quart.left.b <- as.numeric(lapply(trainl, quart.left.b))
#Situation in bottom right quarter
quart.right.b <- function(m){
        nc <- floor(dim(m)[2]/2)
        nr <- floor(dim(m)[1]/2)
        mean(m[(nr+1):dim(m)[1],(nc+1):dim(m)[2]])/(max(m[(nr+1):dim(m)[1],(nc+1):dim(m)[2]]) + 1)
}
featt$quart.right.b <- as.numeric(lapply(trainl, quart.right.b))
featt$label <- as.factor(make.names(featt$label))

#Preparing for model building: data scaling and data partition
set.seed(134)
trainInd <- createDataPartition(y = featt$label,
                                p = 0.8,
                                list = FALSE,
                                times = 1)
training <- featt[trainInd,]
test <- featt[-trainInd,]

preProcValues <-preProcess(training, method = c("center", "scale", "YeoJohnson")) #Potential options for improvements
trainTransformed <- predict(preProcValues, training)
testTransformed <- predict(preProcValues, test)

#Model: 
registerDoMC(10)
fitControl <- trainControl(## 10-fold CV
                        method = "repeatedcv",
                        number = 10,
                        ## repeated ten times
                        #repeats = 10,
                        classProbs = TRUE)
set.seed(343)
Fit <- train(label~.,
          data = trainTransformed,
          method = 'rf',
          metric = 'ROC',
          trControl = fitControl,
          verbose = TRUE,
          allowParallel = TRUE)

prFit <- predict(Fit, testTransformed)
confusionMatrix(testTransformed$label,prFit)
