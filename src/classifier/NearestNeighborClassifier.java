package classifier;

import java.util.List;

import model.ModelClass;
import model.Sample;
import model.SampleWithClass;

public class NearestNeighborClassifier implements Classifier {

	private List<SampleWithClass> trainSamples;
	
	@Override
	public void train(List<SampleWithClass> trainSamples) {
		this.trainSamples = trainSamples;
	}

	@Override
	public ModelClass classify(Sample sample) {
		if (!isTrained()) {
			throw new IllegalStateException("Classifier has to be trained first");
		}
		
		SampleWithClass nearestNeighborSample = findNearestSample(sample);
		return nearestNeighborSample.getModelClass();
	}

	private SampleWithClass findNearestSample(Sample sample) {
		SampleWithClass nearestNeighborSample = null;
		double minimumDistance = Double.MAX_VALUE;
		
		for (SampleWithClass trainSample : trainSamples) {
			double distance = Common.calculateEuclideanDistance(sample, trainSample);
			
			if (distance < minimumDistance) {
				minimumDistance = distance;
				nearestNeighborSample = trainSample;
			}
		}
		
		return nearestNeighborSample;
	}

	@Override
	public boolean isTrained() {
		return trainSamples != null;
	}

}
