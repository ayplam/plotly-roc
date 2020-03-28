### Plotly ROC

Creates interactive ROC or precision-recall curves to simplify evaluation and choose appropriate thresholds. It uses plotly for all graphs and tooltip interactions.

### Requirements
* Python 3.6 or higher
* pandas
* plotly

### Installation
```
git clone git://github.com/seatgeek/fuzzywuzzy.git fuzzywuzzy
cd fuzzywuzzy
python setup.py install
```

### Usage
```
import random
from plotly_roc import metrics, plotly_roc
random.seed(42)
labels = [0]*40 + [1]*40
probas = [random.uniform(0, 0.7) for _ in range(40)] + [random.uniform(0.3, 1) for _ in range(40)]
metrics_df = metrics.metrics_df(labels, probas)
plotly_roc.roc_curve(metrics_df, line_name="Line Title", line_color="tomato", cm_labels=["CAT", "DOG"])
```
![ROC Curve with plotly](roc.gif "ROC Curve with plotly")