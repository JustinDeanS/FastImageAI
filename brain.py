from imageai.Classification import ImageClassification
import os

exec_path = os.getcwd()

prediction = ImageClassification()
# prediction.setModelTypeAsMobileNetV2()
prediction.setModelTypeAsDenseNet121()
# prediction.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth'))
prediction.setModelPath(os.path.join(exec_path, 'densenet121-a639ec97.pth'))
prediction.loadModel()

predctions, probabilities = prediction.classifyImage(os.path.join(exec_path,'cat.jpg'), result_count=5)
for eachPred, eachProb in zip(predctions, probabilities):
    print(f'{eachPred} : {eachProb}')

