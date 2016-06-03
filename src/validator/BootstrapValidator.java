package validator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import classifier.Classifier;
import classifier.MatrixIrreversibleException;
import model.ModelClass;
import model.SampleWithClass;

public class BootstrapValidator implements ClassificationValidator {

	private final int numberOfIterations;
	
	private List<SampleWithClass> trainingSamples = new ArrayList<SampleWithClass>();
	private List<SampleWithClass> testSamples = new ArrayList<SampleWithClass>();
	
	public BootstrapValidator(int numberOfIterations) {
		this.numberOfIterations = numberOfIterations;
	}

	@Override
	public double validate(Classifier classifier, List<SampleWithClass> samples) {
		List<Double> results = new ArrayList<>();
		
		int i = 0;
		while (i < numberOfIterations) {
			try {
				splitSamplesIntoTrainingAndTestSets(samples);
				classifier.train(trainingSamples);
				
				results.add(testClassifier(classifier));
				
				i += 1;
			} catch (MatrixIrreversibleException e) { }
		}
		
		return averageOf(results);
	}

	private void splitSamplesIntoTrainingAndTestSets(List<SampleWithClass> samples) {
		trainingSamples = new ArrayList<>();
		testSamples = new ArrayList<>();
		Set<Integer> usedIndexes = new TreeSet<>();
		Random random = new Random();
		
		for (int i = 0; i < samples.size(); i++) {
			int index = random.nextInt(samples.size());
			trainingSamples.add(samples.get(index));
			usedIndexes.add(index);
		}
		
		for (int index = 0; index < samples.size(); index++) {
			if (!usedIndexes.contains(index)) {
				testSamples.add(samples.get(index));
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

	private double averageOf(List<Double> results) {
		double sum = results
			.stream()
			.mapToDouble(r -> r)
			.sum();
		
		return sum / results.size();
	}

}
