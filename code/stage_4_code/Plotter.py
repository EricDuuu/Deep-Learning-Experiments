import matplotlib.pyplot as plt
def plot_acc_loss(model_name, accuracy, loss):
  plt.style.use("seaborn")
  fig, (ax1,ax2) = plt.subplots(2,sharex=True)
  ax1.set_title(model_name,fontsize=18,fontstyle='italic')

  ax1.plot(accuracy,label="Train Acc",linewidth=2)
  ax1.legend(loc='upper left', shadow=True)
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Accuracy')

  ax2.plot(loss,label="Train Loss",linewidth=2)
  ax2.legend(loc='upper left', shadow=True)
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Loss')

  fig.set_figheight(12.3)
  fig.set_figwidth(10)