import pandas as pd
normal = pd.read_csv("../datasets/Wikipedia simple/normal.aligned", sep = "\t", names=["subject","nr", "sentence"])
simple = pd.read_csv("../datasets/Wikipedia simple/simple.aligned", sep = "\t", names=["subject_simple","nr_simple", "sentence_simple"])

combination = pd.concat([simple,normal],axis=1,join="inner")
#randomize
combination = combination.sample(frac=1,random_state=1)
combination.drop(['nr', 'subject', 'nr_simple', 'subject_simple'],axis=1)

combination.to_csv("../datasets/simplification_classification.csv")