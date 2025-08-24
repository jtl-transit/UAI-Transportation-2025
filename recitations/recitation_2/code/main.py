def main():
    # python
    import biogeme.biogeme as bio
    import biogeme.database as db
    import biogeme.expressions as bioexp
    import pandas as pd

    df = pd.read_csv('/Users/riccardofiorista/Documents/teaching/UAI25/UAI-Transportation-2025/recitations/recitation_2/data/mlogit_Train_wide.csv')
    database = db.Database('db', df)

    # parameters
    beta_price = bioexp.Beta('beta_price', 0, None, None, 0)
    beta_time  = bioexp.Beta('beta_time', 0, None, None, 0)

    # variables
    PRICE = bioexp.Variable('price')
    TIME  = bioexp.Variable('time')
    CHOICE = bioexp.Variable('choiceid')

    # utilities and loglik
    V = {0: beta_price*PRICE + beta_time*TIME, 1: 0}
    logprob = bioexp.bioLogLogit(V, None, CHOICE)
    formulas = {'loglike': logprob}

    biogeme = bio.BIOGEME(database, formulas)
    results = biogeme.estimate()
    print(results.getEstimatedParameters())


if __name__ == "__main__":
    main()
