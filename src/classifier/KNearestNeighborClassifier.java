package classifier;

import static java.util.stream.Collectors.toList;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import model.ModelClass;
import model.Sample;
import model.SampleWithClass;

public class KNearestNeighborClassifier implements Classifier {

	private static final int K = 3;
	
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
		
		List<SampleWithClass> kNearestSamples = findKNearestSamplesTo(sample, K);
		return findTheMostCommonClassOf(kNearestSamples);
	}

	private List<SampleWithClass> findKNearestSamplesTo(Sample sample, int k) {
		List<SampleDistance> result = new ArrayList<>();
		
		for (SampleWithClass trainSample : trainSamples) {
			double euclideanDistance = Common.calculateEuclideanDistance(trainSample, sample);
			SampleDistance sampleDistance = new SampleDistance(trainSample, euclideanDistance);
			result.add(sampleDistance);
		}
		
		return result
			.stream()
			.sorted((a, b) -> Double.compare(a.distance, b.distance))
			.map(s -> s.sample)
			.collect(toList())
			.subList(0, k);
	}
	
	private ModelClass findTheMostCommonClassOf(List<SampleWithClass> samples) {
		Map<ModelClass, Integer> map = new HashMap<>();

	    for (SampleWithClass sample : samples) {
	        Integer val = map.get(sample.getModelClass());
	        map.put(sample.getModelClass(), val == null ? 1 : val + 1);
	    }

	    Entry<ModelClass, Integer> max = null;

	    for (Entry<ModelClass, Integer> e : map.entrySet()) {
	        if (max == null || e.getValue() > max.getValue())
	            max = e;
	    }

	    return max.getKey();
	}

	@Override
	public boolean isTrained() {
		return trainSamples != null;
	}
	
	private static class SampleDistance {
		
		SampleWithClass sample;
		double distance;
		
		SampleDistance(SampleWithClass sample, double distance) {
			this.sample = sample;
			this.distance = distance;
		}
	}

}
