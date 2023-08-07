get.ordinal <- function(position) {
  if (position == "No") {
    return(position)
  }
  # Remove leading zeros if they exist
  position <- gsub("^0+", "", position)
  
  last.two.digits <- as.numeric(substr(position, nchar(position) - 1, nchar(position)))
  
  if (is.na(last.two.digits)) {
    return(position)
  }
  
  if (length(last.two.digits) == 1) {
    if (last.two.digits == 1) {
      ordinal <- paste(position, "st", sep = "")
    } else if (last.two.digits == 2) {
      ordinal <- paste(position, "nd", sep = "")
    } else if (last.two.digits == 3) {
      ordinal <- paste(position, "rd", sep = "")
    } else {
      ordinal <- paste(position, "th", sep = "")
    }
  } else {
    if (last.two.digits >= 11 && last.two.digits <= 13) {
      ordinal <- paste(position, "th", sep = "")
    } else {
      last.digit <- as.numeric(substr(position, nchar(position), nchar(position)))
      if (last.digit == 1) {
        ordinal <- paste(position, "st", sep = "")
      } else if (last.digit == 2) {
        ordinal <- paste(position, "nd", sep = "")
      } else if (last.digit == 3) {
        ordinal <- paste(position, "rd", sep = "")
      } else {
        ordinal <- paste(position, "th", sep = "")
      }
    }
  }
  return(ordinal)
}

transormed.dataframe <- function(dataset) {
  dataset$`Record Label` <- as.factor(dataset$`Record Label`)
  dataset$`Album type` <- as.factor(dataset$`Album type`)
  dataset$`Lead vocal s ` <- as.factor(dataset$`Lead vocal s `)
  dataset$`British charts` <- as.factor(dataset$`British charts`)
  dataset$Certification <- as.factor(dataset$Certification)
  dataset$`Album name` <- as.factor(dataset$`Album name`)
  
  for(i in which(colnames(dataset) == 'UK Peak Pos'):which(colnames(dataset) == 'POL')) {
    variable <- dataset[,i]
    ordinal.positions.i <- sapply(variable, get.ordinal)
    # Get unique positions from the dataframe
    unique.positions.i <- unique(ordinal.positions.i)
    numeric.vector.i <- sort(as.numeric(gsub("[^0-9]", "", unique.positions.i)))
    numeric.vector.i <- sapply(numeric.vector.i, get.ordinal)
    numeric.vector.i <- c("No",numeric.vector.i)
    
    #creating factor variable
    variable.factor <- factor(ordinal.positions.i, levels = numeric.vector.i)
    #adding factor variable 
    dataset[,i] <- variable.factor
  }
  return(dataset)
}

# function for computing evaluation measures
compute.eval.metrics <- function(cmatrix) {
  TP <- cmatrix[1,1] # true positive
  TN <- cmatrix[2,2] # true negative
  FP <- cmatrix[2,1] # false positive
  FN <- cmatrix[1,2] # false negative
  acc <- sum(diag(cmatrix)) / sum(cmatrix)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- 2*precision*recall / (precision + recall)
  c(accuracy = acc, precision = precision, recall = recall, F1 = F1)
}
