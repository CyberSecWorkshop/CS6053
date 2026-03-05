"""
=============================================================
 Week 9 Workshop – Learning from Examples
 COMPLETE Worked Solution – All Tasks, All Algorithms,
 All Datasets
=============================================================
 Files needed in the SAME folder as this script:
   learning4e.py, utils4e.py, probabilistic_learning.py,
   deep_learning.py, test_learning4e.py,
   restaurant.csv, iris.csv, zoo.csv, orings.csv
=============================================================
 Run in Spyder: select a section and press F9, or
 press the green Play button to run everything at once.
=============================================================
"""

import random
from learning4e import (
    DataSet, DecisionTreeLearner, DecisionListLearner,
    NearestNeighborLearner, PluralityLearner,
    err_ratio, train_test_split, information_content
)
from probabilistic_learning import NaiveBayesLearner


# ─────────────────────────────────────────────────────────
# HELPER – print a clean section banner
# ─────────────────────────────────────────────────────────
def banner(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def sub(title):
    print(f"\n--- {title} ---")


# ═════════════════════════════════════════════════════════
# TASK 1 & 2  –  LOADING ALL DATASETS
# ═════════════════════════════════════════════════════════
banner("TASK 1 & 2 – LOADING ALL DATASETS")

# ── Restaurant ──────────────────────────────────────────
restaurant = DataSet(
    name='restaurant',
    attr_names='Alternate Bar Fri/Sat Hungry Patrons Price '
               'Raining Reservation Type WaitEstimate Wait'
)
print("\n[1] Restaurant Dataset")
print(f"    Examples   : {len(restaurant.examples)}")
print(f"    Attributes : {restaurant.attr_names}")
print(f"    Target     : {restaurant.attr_names[restaurant.target]}")
print(f"    Classes    : {restaurant.values[restaurant.target]}")
print("\n    Full problem description:")
print("    ─────────────────────────────────────────────────")
print("    A customer decides whether to WAIT for a table.")
print("    10 input attributes (Patrons, Price, WaitEstimate, etc.)")
print("    Output: Wait = Yes / No")
print("    Training set: 12 hand-labeled examples from the lecture.")
print("\n    All 12 training examples:")
print(f"    {'Alternate':>9} {'Bar':>4} {'Fri':>4} {'Hungry':>7} "
      f"{'Patrons':>8} {'Price':>6} {'Rain':>5} "
      f"{'Res':>4} {'Type':>8} {'Wait-Est':>9} {'Wait':>5}")
print("    " + "-" * 75)
for ex in restaurant.examples:
    print("    " + "  ".join(str(v).rjust(7) for v in ex))

# ── Iris ────────────────────────────────────────────────
iris = DataSet(
    name='iris',
    attr_names='sepal-len sepal-width petal-len petal-width class'
)
print("\n[2] Iris Dataset")
print(f"    Examples   : {len(iris.examples)}")
print(f"    Features   : sepal length, sepal width, petal length, petal width (all in cm)")
print(f"    Classes    : {iris.values[iris.target]}")
print(f"    First row  : {iris.examples[0]}")
print(f"    Last row   : {iris.examples[-1]}")

# ── Zoo ─────────────────────────────────────────────────
zoo = DataSet(
    name='zoo',
    target='type',
    exclude=['name'],
    attr_names='name hair feathers eggs milk airborne aquatic predator '
               'toothed backbone breathes venomous fins legs tail domestic catsize type'
)
print("\n[3] Zoo Dataset")
print(f"    Examples   : {len(zoo.examples)}")
print(f"    Target     : {zoo.attr_names[zoo.target]}")
print(f"    Classes    : {zoo.values[zoo.target]}")
print(f"    Inputs     : 16 Boolean/numeric attributes (hair, feathers, eggs, ...)")

# ── O-Rings ─────────────────────────────────────────────
orings = DataSet(
    name='orings',
    target='Distressed',
    attr_names='Rings Distressed Temp Pressure Flightnum'
)
print("\n[4] O-Rings Dataset (Challenger Space Shuttle)")
print(f"    Examples   : {len(orings.examples)}")
print(f"    Target     : Distressed (number of O-rings under thermal stress)")
print(f"    Context    : Predict O-ring failure risk before a shuttle launch")
print(f"    First row  : {orings.examples[0]}")


# ═════════════════════════════════════════════════════════
# TASK 3 & 4  –  ALGORITHM 1: DECISION TREE
# ═════════════════════════════════════════════════════════
banner("ALGORITHM 1 – DECISION TREE LEARNER")

print("""
HOW IT WORKS:
  The algorithm builds a tree top-down by always asking:
  "Which attribute best separates the Yes cases from the No cases?"

  This is measured using INFORMATION GAIN:
    Gain(attr) = Entropy(parent) - weighted_average_Entropy(children)

  Entropy of a set = -sum( p * log2(p) ) for each class proportion p
    → Entropy = 0   when all examples have the same class (perfect purity)
    → Entropy = 1   when examples are split 50/50 between two classes

  At each node: pick the attribute with the HIGHEST information gain.
  Keep splitting until all leaf nodes are pure (or no attributes left).
""")

# ── Restaurant ──────────────────────────────────────────
sub("Decision Tree on Restaurant Dataset")

# Demonstrate entropy calculation manually
sub("Manual entropy demo")
print("  A set of [6 Yes, 6 No] examples:")
e = information_content([6, 6])
print(f"    Entropy([6 Yes, 6 No]) = {e:.4f}  (maximum uncertainty – 50/50 split)")
print("  A pure set [12 Yes, 0 No]:")
e2 = information_content([12, 0])
print(f"    Entropy([12 Yes, 0 No]) = {e2:.4f}  (zero uncertainty – all same class)")
print("  After splitting on Patrons=Some [6 Yes, 0 No]:")
e3 = information_content([6, 0])
print(f"    Entropy([6 Yes, 0 No]) = {e3:.4f}  → splitting on Patrons removes uncertainty")

sub("Training the tree")
dtl_rest = DecisionTreeLearner(restaurant)
print("\n  Learned Decision Tree structure:")
print("  " + "─" * 50)
dtl_rest.tree.display()
print("  " + "─" * 50)
print("  Observation: 'Patrons' is at the root because it has the")
print("  highest information gain – None always means No,")
print("  Some always means Yes.")

sub("Manual predictions on restaurant")
test_cases = [
    (['Yes','No','No','Yes','Some','$$$','No','Yes','French','0-10'],  'Yes'),
    (['No', 'No','No','No', 'None','$',  'No','No', 'Burger','0-10'],  'No'),
    (['Yes','Yes','Yes','Yes','Full','$$$','No','Yes','Italian','10-30'],'No'),
    (['No', 'Yes','No','Yes','Full','$',  'No','No', 'Burger','>60'],   'No'),
]
print(f"\n  {'Input (Patrons/WaitEst/Type)':>35} {'Expected':>9} {'Got':>9} {'OK?':>4}")
print("  " + "─" * 60)
for feat, expected in test_cases:
    pred = dtl_rest.predict(feat)
    ok = "✓" if pred == expected else "✗"
    label = f"{feat[4]}/{feat[9]}/{feat[8]}"
    print(f"  {label:>35} {expected:>9} {pred:>9} {ok:>4}")

err = err_ratio(dtl_rest, restaurant)
print(f"\n  Training error  : {err:.2%}")
print(f"  Training accuracy: {1-err:.2%}")
print("  (0% error on training data is expected – the tree memorised 12 examples)")

# ── Zoo ─────────────────────────────────────────────────
sub("Decision Tree on Zoo Dataset (80/20 split)")
random.seed(42)
random.shuffle(zoo.examples)  # shuffle before splitting – Zoo is sorted by type
train_zoo, test_zoo = train_test_split(zoo, test_split=0.2)
train_zoo_ds = DataSet(examples=train_zoo, attr_names=zoo.attr_names, target=zoo.target)
test_zoo_ds  = DataSet(examples=test_zoo,  attr_names=zoo.attr_names, target=zoo.target)
dtl_zoo = DecisionTreeLearner(train_zoo_ds)
tr_err = err_ratio(dtl_zoo, train_zoo_ds)
te_err = err_ratio(dtl_zoo, test_zoo_ds)
print(f"  Training accuracy : {1-tr_err:.2%}")
print(f"  Test accuracy     : {1-te_err:.2%}")

sub("Sample zoo predictions")
probes_zoo = [
    ([1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1], 'mammal'),
    ([0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0], 'bird'),
    ([0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0], 'fish'),
]
for feats, expected in probes_zoo:
    pred = dtl_zoo.predict(zoo.sanitize(feats + [expected]))
    ok = "✓" if pred == expected else "✗"
    print(f"  Expected: {expected:>10}   Got: {pred:>10}  {ok}")

# ── Iris ────────────────────────────────────────────────
sub("Decision Tree on Iris Dataset (80/20 split)")
random.seed(42)
random.shuffle(iris.examples)  # shuffle before splitting – Iris is sorted by class
train_iris, test_iris = train_test_split(iris, test_split=0.2)
train_iris_ds = DataSet(examples=train_iris, attr_names=iris.attr_names, target=iris.target)
test_iris_ds  = DataSet(examples=test_iris,  attr_names=iris.attr_names, target=iris.target)
dtl_iris = DecisionTreeLearner(train_iris_ds)
tr_err = err_ratio(dtl_iris, train_iris_ds)
te_err = err_ratio(dtl_iris, test_iris_ds)
print(f"  Training accuracy : {1-tr_err:.2%}")
print(f"  Test accuracy     : {1-te_err:.2%}")

print("\n  First 8 test predictions:")
print(f"  {'Sepal-L':>7} {'Sepal-W':>7} {'Petal-L':>7} {'Petal-W':>7} "
      f"{'Actual':>12} {'Predicted':>12} {'OK?':>4}")
print("  " + "─" * 62)
for ex in test_iris[:8]:
    pred   = dtl_iris.predict(test_iris_ds.sanitize(ex))
    actual = ex[iris.target]
    ok = "✓" if pred == actual else "✗"
    print(f"  {ex[0]:>7} {ex[1]:>7} {ex[2]:>7} {ex[3]:>7} "
          f"{actual:>12} {pred:>12} {ok:>4}")


# ═════════════════════════════════════════════════════════
# TASK 3 & 4  –  ALGORITHM 2: DECISION LIST
# ═════════════════════════════════════════════════════════
banner("ALGORITHM 2 – DECISION LIST LEARNER")

print("""
HOW IT WORKS:
  A Decision List is a simpler, linear version of a Decision Tree.
  Instead of a branching tree, it is a flat ordered list of tests:

    IF  (test_1 passes)  THEN  return outcome_1
    IF  (test_2 passes)  THEN  return outcome_2
    ...
    ELSE                        return default

  The algorithm finds a test that correctly classifies a SUBSET of
  examples, adds it to the list, removes those examples, and repeats.

  Key difference from Decision Tree:
    Decision Tree  → hierarchical branches, re-tests different attributes
    Decision List  → flat ordered rules evaluated top to bottom

  Logical equivalent of the restaurant problem:
    WillWait ⟺ (Patrons=Some)
              ∨ (Patrons=Full ∧ Fri/Sat=True)
              ∨ ...
""")

# Note: DecisionListLearner in learning4e.py has abstract methods
# (find_examples and passes are not implemented in the base class).
# Students need to subclass it or use it as a reference.
# Here we demonstrate how the concept works and show what it produces.

sub("Decision List concept demo on Restaurant")
print("""
  The restaurant training data can be expressed as a decision list:

    Rule 1: IF Patrons = None          THEN Wait = No
    Rule 2: IF Patrons = Some          THEN Wait = Yes
    Rule 3: IF WaitEstimate = >60      THEN Wait = No
    Rule 4: IF WaitEstimate = 0-10     THEN Wait = Yes
    Rule 5: IF Fri/Sat = Yes           THEN Wait = Yes
    Rule 6: IF Hungry = No             THEN Wait = Yes
    Rule 7: IF Alternate = No          THEN Wait = Yes
    Rule 8: ELSE                       THEN Wait = No

  The list is evaluated in order for each new example.
  The FIRST matching rule fires and returns its outcome.

  Compared to the Decision Tree, the Decision List is:
    ✓ Easier to read and explain to non-technical stakeholders
    ✓ Faster to evaluate (linear scan)
    ✗ Potentially less accurate (cannot capture complex interactions)
""")

sub("Comparing Decision Tree vs Decision List (lecture figure)")
print("""
  From the lecture slides, on the restaurant problem:
    Decision Tree  achieves slightly HIGHER accuracy.
    Decision List  achieves slightly LOWER accuracy.

  This is because the restaurant problem has interactions between
  attributes that a tree can capture through branching but a flat
  list cannot.

  For simple, mostly-independent rules, Decision Lists can
  match or outperform Decision Trees.
""")


# ═════════════════════════════════════════════════════════
# TASK 3 & 4  –  ALGORITHM 3: k-NEAREST NEIGHBOURS
# ═════════════════════════════════════════════════════════
banner("ALGORITHM 3 – k-NEAREST NEIGHBOURS (kNN)")

print("""
HOW IT WORKS:
  kNN is a non-parametric method – it stores all training examples
  and does no explicit training step.

  To classify a new example:
    1. Calculate distance from the new example to EVERY training example.
    2. Find the k nearest training examples (smallest distances).
    3. Return the most common class among those k neighbours (majority vote).

  Distance metrics:
    Euclidean  → sqrt(sum((xi - yi)^2))     best for real-valued data
    Manhattan  → sum(|xi - yi|)             good when scales differ
    Hamming    → count of differing values  best for Boolean/categorical

  Choosing k:
    k=1  → memorises training data exactly (overfits)
    k=3  → often a good balance
    k=5  → smoother boundaries, may miss local patterns
    Larger k → lower variance but higher bias
""")

# ── Iris ────────────────────────────────────────────────
sub("kNN on Iris – effect of k")
print(f"  {'k':>4}  {'Train Accuracy':>16}  {'Test Accuracy':>14}")
print("  " + "─" * 38)
random.seed(42)
random.shuffle(iris.examples)  # must shuffle – Iris is sorted by class
train_iris, test_iris = train_test_split(iris, test_split=0.2)
train_iris_ds = DataSet(examples=train_iris, attr_names=iris.attr_names, target=iris.target)
test_iris_ds  = DataSet(examples=test_iris,  attr_names=iris.attr_names, target=iris.target)
for k in [1, 3, 5, 7, 11]:
    model = NearestNeighborLearner(train_iris_ds, k=k)
    tr = 1 - err_ratio(model, train_iris_ds)
    te = 1 - err_ratio(model, test_iris_ds)
    note = " ← perfect (finds itself)" if k == 1 else ""
    print(f"  k={k:<3}  {tr:>15.2%}  {te:>13.2%}{note}")
print("\n  Observation: Training accuracy decreases as k grows.")
print("  Test accuracy is more stable – this is what matters.")

# ── Zoo ─────────────────────────────────────────────────
sub("kNN on Zoo Dataset")
random.seed(42)
random.shuffle(zoo.examples)  # must shuffle – Zoo is sorted by type
train_zoo, test_zoo = train_test_split(zoo, test_split=0.2)
train_zoo_ds = DataSet(examples=train_zoo, attr_names=zoo.attr_names, target=zoo.target)
test_zoo_ds  = DataSet(examples=test_zoo,  attr_names=zoo.attr_names, target=zoo.target)
for k in [1, 3, 5]:
    m = NearestNeighborLearner(train_zoo_ds, k=k)
    tr = 1 - err_ratio(m, train_zoo_ds)
    te = 1 - err_ratio(m, test_zoo_ds)
    print(f"  k={k}  Train: {tr:.2%}  Test: {te:.2%}")

sub("Manual classification probes (Iris)")
knn3 = NearestNeighborLearner(iris, k=3)
probes = [
    ([5.1, 3.5, 1.4, 0.2], 'setosa'),
    ([6.0, 2.9, 4.5, 1.5], 'versicolor'),
    ([7.2, 3.0, 5.8, 1.6], 'virginica'),
]
for feats, expected in probes:
    pred = knn3.predict(feats)
    ok = "✓" if pred == expected else "✗"
    print(f"  {feats}  Expected: {expected:>12}  Got: {pred:>12}  {ok}")


# ═════════════════════════════════════════════════════════
# TASK 3 & 4  –  ALGORITHM 4: NAIVE BAYES
# ═════════════════════════════════════════════════════════
banner("ALGORITHM 4 – NAIVE BAYES")

print("""
HOW IT WORKS:
  Bayes' theorem:
    P(class | features) ∝ P(class) × P(f1|class) × P(f2|class) × ...

  Steps:
    1. Count how often each class appears → P(class)
    2. For each attribute, count value frequencies per class → P(fi|class)
    3. For a new example, compute the product for each class
    4. Return the class with the HIGHEST product

  Why "Naive"?
    It assumes all features are INDEPENDENT given the class.
    This is almost never true in practice, but the algorithm still
    performs well because the relative ordering of class probabilities
    is usually preserved even when independence is violated.

  Two modes:
    Discrete   → counts exact occurrences; good for categorical data
    Continuous → fits a Gaussian curve per feature per class;
                 good for real-valued data (use this for Iris)
""")

# ── Restaurant – discrete ────────────────────────────────
sub("Naive Bayes (Discrete) on Restaurant")
nb_rest = NaiveBayesLearner(restaurant, continuous=False)
err_r = err_ratio(nb_rest, restaurant)
print(f"  Training error   : {err_r:.2%}")
print(f"  Training accuracy: {1-err_r:.2%}")

# ── Iris – discrete vs continuous ────────────────────────
sub("Naive Bayes on Iris – Discrete vs Continuous")
random.seed(42)
random.shuffle(iris.examples)  # must shuffle – Iris is sorted by class
train_iris, test_iris = train_test_split(iris, test_split=0.2)
train_iris_ds = DataSet(examples=train_iris, attr_names=iris.attr_names, target=iris.target)
test_iris_ds  = DataSet(examples=test_iris,  attr_names=iris.attr_names, target=iris.target)

nb_d = NaiveBayesLearner(train_iris_ds, continuous=False)
nb_c = NaiveBayesLearner(train_iris_ds, continuous=True)
print(f"\n  {'Mode':<30} {'Train Acc':>10} {'Test Acc':>10}")
print("  " + "─" * 52)
for label, model in [("Discrete", nb_d), ("Continuous (Gaussian)", nb_c)]:
    tr = 1 - err_ratio(model, train_iris_ds)
    te = 1 - err_ratio(model, test_iris_ds)
    print(f"  {label:<30} {tr:>10.2%} {te:>10.2%}")
print("\n  Continuous outperforms Discrete on Iris because petal/sepal")
print("  measurements are real-valued – exact value matches fail too often.")

# ── Zoo – discrete ───────────────────────────────────────
sub("Naive Bayes (Discrete) on Zoo")
random.seed(42)
random.shuffle(zoo.examples)  # must shuffle – Zoo is sorted by type
train_zoo, test_zoo = train_test_split(zoo, test_split=0.2)
train_zoo_ds = DataSet(examples=train_zoo, attr_names=zoo.attr_names, target=zoo.target)
test_zoo_ds  = DataSet(examples=test_zoo,  attr_names=zoo.attr_names, target=zoo.target)
nb_zoo = NaiveBayesLearner(train_zoo_ds, continuous=False)
tr = 1 - err_ratio(nb_zoo, train_zoo_ds)
te = 1 - err_ratio(nb_zoo, test_zoo_ds)
print(f"  Train accuracy: {tr:.2%}   Test accuracy: {te:.2%}")
print("  Zoo attributes are Boolean → Discrete mode is appropriate here.")

sub("Manual classification probes (Iris Continuous)")
nb_iris_c = NaiveBayesLearner(iris, continuous=True)
for feats, expected in probes:
    pred = nb_iris_c(feats)
    ok = "✓" if pred == expected else "✗"
    print(f"  {feats}  Expected: {expected:>12}  Got: {pred:>12}  {ok}")


# ═════════════════════════════════════════════════════════
# TASK 5  –  FULL ALGORITHM COMPARISON
# ═════════════════════════════════════════════════════════
banner("TASK 5 – ALGORITHM COMPARISON ACROSS ALL DATASETS")

print("\nUsing fixed random seed (42) for reproducibility.")
print("All datasets split 80% training / 20% test.\n")

# Use copies so repeated shuffles in earlier sections don't affect Task 5
import copy as _copy

datasets = {
    'Restaurant (12 examples)': (restaurant, False),  # (dataset, is_numeric)
    'Iris (150 examples)':      (iris,       True),
    'Zoo (101 examples)':       (zoo,        False),
}

for ds_name, (dataset, is_numeric) in datasets.items():
    print(f"\n{'─'*65}")
    print(f"  DATASET: {ds_name}")
    print(f"{'─'*65}")

    # Work on a fresh copy so shuffles here don't bleed into each other
    ds_copy = _copy.copy(dataset)
    ds_copy.examples = list(dataset.examples)

    random.seed(42)
    if len(ds_copy.examples) < 20:
        # Too small to split – train and test on same data (restaurant)
        train_ds = ds_copy
        test_ds  = ds_copy
        note = " (train=test – only 12 examples, too small to split)"
    else:
        random.shuffle(ds_copy.examples)
        tr_ex, te_ex = train_test_split(ds_copy, test_split=0.2)
        train_ds = DataSet(examples=tr_ex, attr_names=ds_copy.attr_names, target=ds_copy.target)
        test_ds  = DataSet(examples=te_ex, attr_names=ds_copy.attr_names, target=ds_copy.target)
        note = f" ({len(tr_ex)} train / {len(te_ex)} test)"

    print(f"  Split{note}\n")

    # Build algorithm list – skip Continuous NB for categorical datasets
    algorithms = [
        ("Plurality (baseline)",   PluralityLearner(train_ds)),
        ("Decision Tree",          DecisionTreeLearner(train_ds)),
        ("kNN (k=1)",              NearestNeighborLearner(train_ds, k=1)),
        ("kNN (k=3)",              NearestNeighborLearner(train_ds, k=3)),
        ("kNN (k=5)",              NearestNeighborLearner(train_ds, k=5)),
        ("Naive Bayes (Discrete)", NaiveBayesLearner(train_ds, continuous=False)),
    ]
    if is_numeric:
        # Continuous (Gaussian) NB only makes sense for real-valued attributes
        algorithms.append(
            ("Naive Bayes (Continuous)", NaiveBayesLearner(train_ds, continuous=True))
        )
    else:
        algorithms.append(
            ("Naive Bayes (Continuous)", None)   # not applicable
        )

    print(f"  {'Algorithm':<30} {'Train Acc':>10} {'Test Acc':>10}")
    print("  " + "─" * 52)
    for name, model in algorithms:
        if model is None:
            print(f"  {name:<30} {'N/A':>10} {'N/A':>10}  (categorical data – use Discrete instead)")
            continue
        tr_err = err_ratio(model, train_ds)
        te_err = err_ratio(model, test_ds)
        print(f"  {name:<30} {1-tr_err:>9.1%} {1-te_err:>9.1%}")


# ─────────────────────────────────────────────────────────
# O-RINGS NOTE
# ─────────────────────────────────────────────────────────
banner("NOTE – O-RINGS DATASET")
print("""
  The O-Rings dataset predicts how many O-rings are under thermal
  stress during a Space Shuttle launch (numeric output, 0–6).

  It is a REGRESSION problem (continuous output), not classification.
  The algorithms covered in this workshop are classifiers.

  To use O-rings in a classification context you would need to
  discretise the output (e.g. Distressed > 0 → Yes, else No).

  It is included in learning4e.py as a benchmark dataset and is
  used in the textbook's linear regression examples (Chapter 19).

  For this workshop it serves as an illustration that not all
  datasets suit the same algorithm family.
""")
orings_binary = DataSet(
    name='orings',
    target='Distressed',
    attr_names='Rings Distressed Temp Pressure Flightnum'
)
# Show the data for awareness
print("  O-rings data (first 5 rows):")
print(f"  {'Rings':>6} {'Distressed':>11} {'Temp(F)':>8} {'Pressure':>9} {'FlightNum':>10}")
print("  " + "─" * 48)
for ex in orings_binary.examples[:5]:
    print("  " + "  ".join(str(v).rjust(9) for v in ex))


banner("Workshop Complete – All Tasks Executed")
print("""
  Summary of what was covered:
    Task 1 & 2  : Loaded Restaurant, Iris, Zoo, O-Rings datasets
    Algorithm 1 : Decision Tree  (Restaurant, Zoo, Iris)
    Algorithm 2 : Decision List  (concept explanation + comparison)
    Algorithm 3 : k-Nearest Neighbours  (Iris, Zoo) with k comparison
    Algorithm 4 : Naive Bayes  (Discrete and Continuous modes)
    Task 5      : Full comparison table across all datasets
""")
