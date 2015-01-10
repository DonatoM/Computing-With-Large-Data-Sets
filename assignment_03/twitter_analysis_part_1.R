library(plyr)
library("openNLP")

pause <- function ()
{
  cat("Pause. Press <Enter> to continue...")
  readline()
  invisible()
}

Ebola = read.csv('Ebola_frequencies.csv')
MichaelBrown = read.csv('IfTheyGunnedMeDown_frequencies.csv')
Cities = read.csv('USTop10Cities_frequencies.csv')
WordValance = read.csv('AFINN-111.txt', sep="\t", header=FALSE, col.names=c("word","polarity"))

# Separate by weeks.
Ebola = rename(Ebola, c("X2014.10.06"="w1", "X2014.10.13"="w2", "X2014.10.20"="w3", "X2014.10.27"="w4"))
MichaelBrown = rename(MichaelBrown, c("X2014.10.01"="w1", "X2014.10.08"="w2", "X2014.10.15"="w3", "X2014.10.22"="w4"))
Cities = rename(Cities, c("X2014.10.31"="w1", "X2014.11.07"="w2", "X2014.11.14"="w3", "X2014.11.21"="w4"))


get_sentiment <- function(x) {
  data = WordValance[x,]$polarity
  if (length(is.na(data)) > 1) {
    print("LENGTH WAS > 1")
    print(length(is.na(data)))
  }
  if (is.na(data))
    return(0)
  else
    return(data)
}

sentiments <- function(dataSet) {
  dataSet$sentiment = apply(dataSet[c("term")], 1, get_sentiment)
  dataSet$s1 = apply(dataSet[c("sentiment", "w1")], 1, prod)
  dataSet$s2 = apply(dataSet[c("sentiment", "w2")], 1, prod)
  dataSet$s3 = apply(dataSet[c("sentiment", "w3")], 1, prod)
  dataSet$s4 = apply(dataSet[c("sentiment", "w4")], 1, prod)
  return(dataSet)
}


Ebola = sentiments(Ebola)
MichaelBrown = sentiments(MichaelBrown)
Cities = sentiments(Cities)

# histogram of sentiment of each token where sentiment is != 0
# not considering frequency of each token
hist(MichaelBrown[which(!MichaelBrown$sentiment == 0),]$sentiment,xlab="Value",main="Sentiment Values For Michael Brown Tweets")
pause()
hist(Cities[which(!Cities$sentiment == 0),]$sentiment,xlab="Value",main="Sentiment Values For City Tweets")
pause()
hist(Ebola[which(!Ebola$sentiment == 0),]$sentiment,xlab="Value",main="Sentiment Values For Ebola Tweets")

