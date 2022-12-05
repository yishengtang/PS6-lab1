import numpy as np
import scipy.stats as stats
import patsy
import sklearn.linear_model as linear
import random
import pandas as pd
from pprint import pprint

# we're not currently using this because the LaTeX
# experience is so awful (it doesn't use HTML but
# the plaintext representation).
from IPython.display import HTML, display_html
from tabulate import tabulate

ALGORITHMS = {
    "linear": linear.LinearRegression,
    "ridge": linear.Ridge,
    "lasso": linear.Lasso
}

def summarize(formula, X, y, model, style='linear'):
    result = {}
    result["formula"] = formula
    result["n"] = len(y)
    result["model"] = model
    # I think this is a bug in Scikit Learn 
    # because lasso should work with multiple targets.
    if style == "lasso":
        result["coefficients"] = model.coef_
    else:
        result["coefficients"] =  model.coef_[0]
    result["r_squared"] = model.score( X, y)
    y_hat = model.predict(X)
    result["residuals"] = y - y_hat
    result["y_hat"] = y_hat
    result["y"]  = y
    sum_squared_error = sum([e**2 for e in result[ "residuals"]])[0]

    n = len(result["residuals"])
    k = len(result["coefficients"])
    
    result["sigma"] = np.sqrt( sum_squared_error / (n - k))
    return result

def linear_regression(formula, data=None, style="linear", params={}):
    if data is None:
        raise ValueError( "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    params["fit_intercept"] = False

    y, X = patsy.dmatrices(formula, data, return_type="matrix")
    algorithm = ALGORITHMS[style]
    algo = algorithm(**params)
    model = algo.fit( X, y)

    result = summarize(formula, X, y, model, style)

    return result

def logistic( z):
    return 1.0 / (1.0 + np.exp( -z))

def logistic_regression( formula, data=None):
    if data is None:
        raise ValueError( "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    result = {}
    result[ "formula"] = formula
    result[ "n"] = data.shape[ 0]

    y, X = patsy.dmatrices( formula, data, return_type="matrix")
    y = np.ravel( y) # not sure why this is needed for LogisticRegression but not LinearRegression

    model = linear.LogisticRegression( fit_intercept=False).fit( X, y)
    result["model"] = model

    result[ "coefficients"] = model.coef_[ 0]

    y_hat = model.predict( X)
    result[ "residuals"] = y - y_hat
    result["y_hat"] = y_hat 
    result["y"] = y

    # efron's pseudo R^2
    y_bar = np.mean(y)
    pr = model.predict_proba(X).transpose()[1]
    result["probabilities"] = pr
    efrons_numerator = np.sum((y - pr)**2) 
    efrons_denominator = np.sum((y-y_bar)**2)
    result["r_squared"] = 1 - (efrons_numerator/efrons_denominator)

    # error rate
    result["sigma"] = np.sum(np.abs(result["residuals"]))/result["n"]*100

    n = len( result[ "residuals"])
    k = len( result[ "coefficients"])

    return result

def bootstrap_linear_regression( formula, data=None, samples=100, style="linear", params={}):
    if data is None:
        raise ValueError( "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
    
    bootstrap_results = {}
    bootstrap_results[ "formula"] = formula

    variables = [x.strip() for x in formula.split("~")[1].split( "+")]
    variables = ["intercept"] + variables
    bootstrap_results[ "variables"] = variables
    
    coeffs = []
    sigmas = []
    rs = []

    n = len(data)
    bootstrap_results[ "n"] = n
    
    for i in range( samples):
        sampling = data.sample(len(data), replace=True)
        results = linear_regression( formula, data=sampling, style=style, params=params)
        coeffs.append( results[ "coefficients"])
        sigmas.append( results[ "sigma"])
        rs.append( results[ "r_squared"])
    
    coeffs = pd.DataFrame( coeffs, columns=variables)
    sigmas = pd.Series( sigmas, name="sigma")
    rs = pd.Series( rs, name="r_squared")

    bootstrap_results[ "resampled_coefficients"] = coeffs
    bootstrap_results[ "resampled_sigma"] = sigmas
    bootstrap_results[ "resampled_r^2"] = rs
    
    result = linear_regression( formula, data=data)
    
    bootstrap_results[ "residuals"] = result[ "residuals"]
    bootstrap_results[ "coefficients"] = result[ "coefficients"]
    bootstrap_results[ "sigma"] = result[ "sigma"]
    bootstrap_results[ "r_squared"] = result[ "r_squared"]
    bootstrap_results["model"] = result["model"]
    bootstrap_results["y"] = result["y"]
    bootstrap_results["y_hat"] = result["y_hat"]
    return bootstrap_results

def bootstrap_logistic_regression( formula, data=None, samples=100):
    if data is None:
        raise ValueError( "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
    
    bootstrap_results = {}
    bootstrap_results[ "formula"] = formula

    variables = [x.strip() for x in formula.split("~")[1].split( "+")]
    variables = ["intercept"] + variables
    bootstrap_results[ "variables"] = variables
    
    coeffs = []
    sigmas = []
    rs = []

#    n = data.shape[ 0]
    n = len(data)
    bootstrap_results[ "n"] = n
    
    for i in range( samples):
        sampling = data.sample(n, replace=True)
        results = logistic_regression( formula, data=sampling)
        coeffs.append( results[ "coefficients"])
        sigmas.append( results[ "sigma"])
        rs.append( results[ "r_squared"])
    
    coeffs = pd.DataFrame( coeffs, columns=variables)
    sigmas = pd.Series( sigmas, name="sigma")
    rs = pd.Series( rs, name="r_squared")

    bootstrap_results[ "resampled_coefficients"] = coeffs
    bootstrap_results[ "resampled_sigma"] = sigmas
    bootstrap_results[ "resampled_r^2"] = rs
    
    result = logistic_regression( formula, data=data)
    
    bootstrap_results[ "residuals"] = result[ "residuals"]
    bootstrap_results[ "coefficients"] = result[ "coefficients"]
    bootstrap_results[ "sigma"] = result[ "sigma"]
    bootstrap_results[ "r_squared"] = result[ "r_squared"]
    bootstrap_results["model"] = result["model"]
    return bootstrap_results

def fmt(n, sd=2):
    return (r"{0:." + str(sd) + "f}").format(n)


def boldify(xs, format):
    if format == "html":
        return ["<strong>" + x + "</strong>" if x != "" else "" for x in xs]
    if format == "markdown":
        return ["**" + x + "**" if x != "" else "" for x in xs]
    # latex
    return ["\\textbf{" + x + "}" if x != "" else "" for x in xs]


def results_table(fit, sd=2,bootstrap=False, is_logistic=False, format="html"):
    result = {} 
    result["model"] = [fit["formula"]]

    variables = [v.strip() for v in [""] + fit["formula"].split("~")[1].split( "+")]
    if format == 'latex':
        variables = [v.replace("_", "\\_") for v in variables]
    coefficients = [] 

    if bootstrap:
        bounds = fit[ "resampled_coefficients"].quantile([0.025, 0.975])
        bounds = bounds.transpose()
        bounds = bounds.values.tolist()
        for i, b in enumerate(zip(variables, fit["coefficients"], bounds)):
            coefficient = [b[0], f"$\\beta_{{{i}}}$", fmt(b[1], sd), fmt(b[2][0], sd), fmt(b[2][1], sd)]
            if is_logistic:
                if i == 0:
                    coefficient.append(fmt(logistic(b[1]), sd))
                else:
                    coefficient.append(fmt(b[1]/4, sd))
            coefficients.append(coefficient)
    else:
        for i, b in enumerate(zip(variables, fit["coefficients"])):
            coefficients.append([b[0], f"$\\beta_{{{i}}}$", fmt(b[1], sd)])
    result["coefficients"] = coefficients

    error = r"$\sigma$"
    r_label = r"$R^2$"
    if is_logistic:
        error = "Error (%)"
        r_label = r"Efron's $R^2$"
    if bootstrap:
        sigma_bounds = stats.mstats.mquantiles( fit[ "resampled_sigma"], [0.025, 0.975])
        r_bounds = stats.mstats.mquantiles( fit[ "resampled_r^2"], [0.025, 0.975])
        metrics = [
            [error, fmt(fit["sigma"], sd), fmt(sigma_bounds[0], sd), fmt(sigma_bounds[1], sd)], 
            [r_label, fmt(fit["r_squared"], sd), fmt(r_bounds[0], sd), fmt(r_bounds[1], sd)]]
    else:
        metrics = [
            [error, fmt(fit["sigma"], sd)], 
            [r_label, fmt(fit["r_squared"], sd)]]

    result["metrics"] = metrics

    title = f"Model: {result['model'][0]}"
    rows = []
    if bootstrap:
        rows.append(boldify(["", "", "", "95% BCI"], format))
    if is_logistic:
        if bootstrap:
            header = boldify(["Coefficients", "", "Mean", "Lo", "Hi", "P(y=1)"], format)
        else:
            header = boldify(["Coefficients", "", "Value"], format)
    else:
        if bootstrap:
            header = boldify(["Coefficients", "", "Mean", "Lo", "Hi"], format)
        else:
            header = boldify(["Coefficients", "", "Value"], format)
    rows.append(header)

    for row in result["coefficients"]:
        rows.append(row)

    rows.append([])

    if bootstrap:
        rows.append(boldify(["Metrics", "Mean", "Lo", "Hi"], format))
    else:
        rows.append(boldify(["Metrics", "Value"], format))
    for row in result["metrics"]:
        rows.append(row)

    return title, rows


class ResultsWrapper(object):
    def __init__(self, fit, sd=2, bootstrap=False, is_logistic=False):
        self.fit = fit
        self.sd = sd
        self.bootstrap = bootstrap
        self.is_logistic = is_logistic

    def _repr_markdown_(self):
        title, table = results_table(self.fit, self.sd, self.bootstrap, self.is_logistic, format="markdown")
        table = tabulate(table, tablefmt="github")
        markdown = title + "\n" + table
        return markdown

    def _repr_html_(self):
        title, table = results_table(self.fit, self.sd, self.bootstrap, self.is_logistic, format="html")
        table = tabulate(table, tablefmt="html")
        table = table.replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong")
        return f"<p><strong>{title}</strong><br/>{table}</p>"

    def _repr_latex_(self):
        title, table = results_table(self.fit, self.sd, self.bootstrap, self.is_logistic, format="latex")

        title = title.replace("~", "$\\sim$").replace("_", "\\_")

        table = tabulate(table, tablefmt="latex_booktabs")
        table = table.replace("textbackslash{}", "").replace("\^{}", "^").replace("\_", "_")
        table = table.replace("\\$", "$").replace("\\{", "{").replace("\\}", "}")
        latex = "\\textbf{" + title + "}\n\n" + table
        return latex


def print_csv(table):
    print("Linear Regression")
    print("Coefficients")
    for item in table["coefficients"]:
        print(','.join(item))
    print("Metrics")
    for item in table["metrics"]:
        print(','.join(item))
    
def simple_describe_lr(fit, sd=2):
    return ResultsWrapper(fit, sd)

def simple_describe_lgr(fit, sd=2):
    return ResultsWrapper(fit, sd, False, True)

def describe_bootstrap_lr(fit, sd=2):
    return ResultsWrapper(fit, sd, True, False)

def describe_bootstrap_lgr(fit, sd=2):
    return ResultsWrapper(fit, sd, True, True)

def strength(pr):
    if 0 <= pr <= 0.33:
        return "weak"
    if 0.33 < pr <= 0.66:
        return "mixed"
    return "strong"

# {"var1": "+", "var2": "-"}
def evaluate_coefficient_predictions(predictions, result):
    coefficients = result["resampled_coefficients"].columns
    for coefficient in coefficients:
        if coefficient == 'intercept':
            continue
        if predictions[coefficient] == '+':
            pr = np.mean(result["resampled_coefficients"][coefficient] > 0)
            print("{0} P(>0)={1:.3f} ({2})".format(coefficient, pr, strength(pr)))
        else:
            pr = np.mean(result["resampled_coefficients"][coefficient] < 0)
            print("{0} P(<0)={1:.3f} ({2})".format(coefficient, pr, strength(pr)))

def adjusted_r_squared(result):
    adjustment = (result["n"] - 1)/(result["n"] - len(result["coefficients"]) - 1 - 1)
    return 1 - (1 - result["r_squared"]) * adjustment