package validator;

import java.util.ArrayList;
import java.util.List;

import model.ModelClass;
import model.SampleWithClass;
import classifier.Classifier;
import classifier.MatrixIrreversibleException;

public class SimpleValidator implements ClassificationValidator {

	private final int trainingSetSizeInPercents;
	
	private List<SampleWithClass> trainingSamples = new ArrayList<>();
	private List<SampleWithClass> testSamples = new ArrayList<>();
	
	public SimpleValidator(int trainingSetSizeInPercents) {
		this.trainingSetSizeInPercents = trainingSetSizeInPercents;
	}

	@Override
	public double validate(Classifier classifier, List<SampleWithClass> samples) {
		while (true) {
			try {
				splitSamplesIntoTrainingAndTestSets(samples);
				classifier.train(trainingSamples);
				
				return testClassifier(classifier);
			} catch (MatrixIrreversibleException e) { }
		}
	}

	private void splitSamplesIntoTrainingAndTestSets(List<SampleWithClass> samples) {
    	double trainingSetBelongingProbability = trainingSetSizeInPercents / 100.0;
    	
    	// uncomment following line if you wanna get better results
    	// because "the order does matter"
    	// Collections.shuffle(samples);
    	
    	for (SampleWithClass sample : samples) {
			if (Math.random() <= trainingSetBelongingProbability) {
				trainingSamples.add(sample);
			} else {
				testSamples.add(sample);
			}
		}
    }

	private double testClassifier(Classifier classifier) {
		int numberOfValidClassifications = 0;
		
		for (SampleWithClass sample : testSamples) {
			ModelClass modelClass = classifier.classify(sample);
			if (modelClass.equals(sample.getModelClass())) {
				numberOfValidClassifications += 1;
			}
		}
		
		return (double)numberOfValidClassifications / testSamples.size();
	}
		
}
