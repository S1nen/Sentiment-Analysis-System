from data_preprocessing2 import x_train_scaled,y_train,x_test_scaled,y_test
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier 
import pickle

xgb_model=XGBClassifier()
xgb_model.fit(x_train_scaled,y_train)
print("accuracy on training data :",xgb_model.score(x_train_scaled,y_train))
print("accuracy on test data :",xgb_model.score(x_test_scaled,y_test))

#found accuracy on trainig data-0.92
#found accuracy on test data-0.84

y_pred3=xgb_model.predict(x_test_scaled)
print("classification report:",classification_report(y_test,y_pred3))

cm=confusion_matrix(y_test,y_pred3)
cm_display=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=xgb_model.classes_,)
cm_display.plot()
plt.show()

pickle.dump(xgb_model,open("models/xgb_model2.pkl","wb"))