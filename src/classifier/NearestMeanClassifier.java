package classifier;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import model.ModelClass;
import model.Sample;
import model.SampleWithClass;
import Jama.Matrix;

public class NearestMeanClassifier implements Classifier {

	private Map<ModelClass, ClassStatisticData> classesStatisticData;
	
	@Override
	public void train(List<SampleWithClass> trainSamples) {
		classesStatisticData = calculateStatisticDataForClasses(trainSamples);
	}

	private Map<ModelClass, ClassStatisticData> calculateStatisticDataForClasses(List<SampleWithClass> trainSamples) {
		Map<ModelClass, List<SampleWithClass>> samplesGroupedByClass = groupSamplesByClass(trainSamples);
		Map<ModelClass, ClassStatisticData> classesStatisticData = new HashMap<>();
		
		for (ModelClass modelClass : samplesGroupedByClass.keySet()) {
			List<SampleWithClass> samplesOfClass = samplesGroupedByClass.get(modelClass);
			ClassStatisticData classStatisticData = calculateClassStatisticDataFrom(samplesOfClass);
			classesStatisticData.put(modelClass, classStatisticData);
		}
		
		return classesStatisticData;
	}

	private Map<ModelClass, List<SampleWithClass>> groupSamplesByClass(
			List<SampleWithClass> trainSamples) {
		return trainSamples
				.stream()
				.collect(Collectors.groupingBy(SampleWithClass::getModelClass));
	}
	
	private ClassStatisticData calculateClassStatisticDataFrom(List<SampleWithClass> samplesOfClass) {
		Matrix mean = Common.calculateMean(samplesOfClass);	
		Matrix covarianceMatrix = Common.calculateCovarianceMatrix(samplesOfClass, mean);
		
		return new ClassStatisticData(mean, covarianceMatrix);
	}
	
	@Override
	public ModelClass classify(Sample sample) {
		if (!isTrained()) {
			throw new IllegalStateException("Classifier has to be trained first");
		}
		
		double minimalMachalonobisDistance = Double.MAX_VALUE;
		ModelClass nearestMeanClass = null;
		
		for (Entry<ModelClass, ClassStatisticData> entry : classesStatisticData.entrySet()) {
			double machalonobisDistance = Common.calculateMahalonobisDistance(sample, entry.getValue());
			if (machalonobisDistance < minimalMachalonobisDistance) {
				minimalMachalonobisDistance = machalonobisDistance;
				nearestMeanClass = entry.getKey();
			}
		}
		
		return nearestMeanClass;
	}

	@Override
	public boolean isTrained() {
		return classesStatisticData != null;
	}
	
}
