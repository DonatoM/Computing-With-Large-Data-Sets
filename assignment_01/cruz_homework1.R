pause <- function ()
{
  cat("Pause. Press <Enter> to continue...")
  readline()
  invisible()
}

histogram <- function(){
  bg.surv <- read.csv(file = "big-data-survey-2014-fall-interests.csv")

  # Created an empty list. I will store all correlation values.
  cor_values = list()
  # Create a variable that represents my data.
  me <- as.numeric(bg.surv[28,8:21])

  for(i in 1:42){
    student <- as.numeric(bg.surv[i,8:21])
    # cor() returns a value of 1 if the two elements compared are identical. Hence, we need
    # to ignore this portion of the data to avoid inconsistent models.
    if(cor(me,student) == 1){next}
    cor_values[i]= cor(me,student)
  }

  # Got rid of the NULL value, which resulted from the next statement in the for loop above.
  cor_values[sapply(cor_values, is.null)] <- NULL
  hist(as.numeric(cor_values))
  index_of_max <- which.max(cor_values)
  index_of_min <- which.min(cor_values)
  print(paste0("Person that I am most similar with is: ", bg.surv[index_of_max,1]))
  print(paste0("Person that I am most dissimilar with is: ", bg.surv[index_of_min,1]))
}

distance <- function(){
  bg.surv <- read.csv(file = "big-data-survey-2014-fall-interests.csv")

  # Created an empty list. I will store all distance values.
  dist_values = list()
  # Create a variable that represents my data.
  me <- as.numeric(bg.surv[28,8:21])

  for(i in 1:42){
    student <- as.numeric(bg.surv[i,8:21])
    # Again, here I check if the current person is me.
    if(i == 28){next}
    dist_values[i]= dist(rbind(me,student))

  }

  # Got rid of the NULL value because it will cause inconsistency in the Histogram.
  dist_values[sapply(dist_values, is.null)] <- NULL
  hist(as.numeric(dist_values))
  index_of_max <- which.max(dist_values)
  index_of_min <- which.min(dist_values)
  print(paste0("Person that I am most similar with is: ", bg.surv[index_of_max,1]))
  print(paste0("Person that I am most dissimilar with is: ", bg.surv[index_of_min,1]))
}

pause()
histogram()
pause()
distance()
