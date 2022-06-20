
from sklearn.metrics import accuracy_score
from OFS import ofs_partial
from data_loader import load_magic04
import os

from data_loader import load_svmguide3

os.chdir('/Users/armanr/Documents/phd/ODT/Continuous/feature_selection')


# X, Y = load_svmguide3()
X, Y = load_magic04()
predictions, num_mistakes = ofs_partial(R=10, eps=0.2, stepsize=0.2, num_features=int(0.1*X.shape[1]), X=X, Y=Y)

print('accuracy: ' + str(accuracy_score(Y, predictions)))
print('number of mistakes: ' + str(num_mistakes))



