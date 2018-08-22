\name{DiscNoise}
\alias{DiscNoise}
\title{Impact of Discretization Noise on Machine Learning Classifiers when Studying Software Engineering Datasets}
\description{
This package is a framework for analyzing the impact of discretization noise (i.e., The noise generated during the discretization of a continuous dependent variable.) on the performance and interpretation of a classifier.
}
\details{
This package's primary function that invokes the framework and computes the impact of discretization noise (found in the demarcated noisy area)

1) It provides both the performance and interpretation impact. The function required to compute this can be invoked by calling \code{compute_impact} with suitable parameters

2) The datasets used in the paper are provided as csv files in the \code{\link{https://github.com/rgopikrishnan91/DiscNoise/data}} folder of the github repository.
For all the functions in the package see \code{help(package = "distcompare")}
}
\author{
Maintainer: Gopi Krishnan Rajbahadur \url{http://gopikrishnanrajbahadur.me}
}
\references{
Rajbahadur, G. K., Wang, S., Kamei, Y., & Hassan, A. E. (2018). Impact of Discretization Noise on Machine Learning Classifiers when Studying Software Engineering Datasets. TBA.
 }