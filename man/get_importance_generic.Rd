% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DiscNoise.R
\name{get_importance_generic}
\alias{get_importance_generic}
\title{Function for generating the importance values of the classifiers}
\usage{
get_importance_generic(model, var)
}
\arguments{
\item{model}{a r model. A trained classifier model whose importance is to be evaluated}

\item{var}{a character vector, containing the columnames of the test data set}
}
\value{
a numeric vector, with all the importance values for the features involved
}
\description{
Function for generating the importance values of the classifiers
}
