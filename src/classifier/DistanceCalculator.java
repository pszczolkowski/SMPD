package classifier;

import model.Sample;

public interface DistanceCalculator {

	double calculate(Sample sample, ClassStatisticData classStatisticData);
	
}
