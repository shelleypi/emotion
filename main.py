from emotion_recognition_deep import DeepEmotionRecognizer
from emotion_recognition import EmotionRecognizer, plot_model_comparisons
from data_preparation import prepare_data

if __name__ == '__main__':
    deep_rec = DeepEmotionRecognizer(override=False, cnn_only=True, n_cnn_layers=2, n_dense_layers=0)
    deep_rec.train()
    deep_rec.plot_loss_and_acc()
    deep_rec.train_score()
    deep_rec.val_score()
    deep_rec.plot_confusion_matrix('val')

    # deep_rec.predict()
    # deep_rec.plot_confusion_matrix('test')

#################################################################################

    # rec = EmotionRecognizer(model=None, override=False)
    # rec.train()
    # rec.plot_confusion_matrix('val')
    # plot_model_comparisons()
