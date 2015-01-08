assignment <- function (){
	library(lasso2)
	load("baa.ratios.rda")
	data <- read.table("baa.TFs.txt",stringsAsFactors=FALSE)
	# Potential Predictors based on HW description.
	predictors <- data$V1
	# Created matrix.
	initial_matrix<-t(ratios)
	# Created list.
	final <- list()
	predictors_list = length(predictors)
	matrix_column_nums = length(colnames(initial_matrix))
	highest <-Inf
	lowest<-Inf

	for(i in 1:predictors_list){
		for (j in 1:matrix_column_nums){
			if(predictors[i]==colnames(initial_matrix)[j]){
				final[[i]] = initial_matrix[,j]
			}
		}
	}

	# Remove NULL elements from list.
	final[sapply(final,is.null)]<-NULL
	# Combine list.
	gen_matrix = do.call(cbind, final)
	# Generate sequences.
	range <- seq(.01,1.00,.01)

	for(i in range){
		gen_model <- l1ce(ratios[1,]~gen_matrix,standardize=TRUE,a=FALSE,sweep.out=~1,bound=i)
		avg <- mean(residuals(gen_model)**2)
		if(!is.nan(avg)){
			if(avg<lowest){
				lowest <-avg
				highest <- i
			}
		}
	}
	opt.model <- l1ce(ratios[1,]~gen_matrix,standardize=TRUE,a=FALSE,sweep.out=~1,bound = highest)
	highest
	opt.model
}
assignment()

