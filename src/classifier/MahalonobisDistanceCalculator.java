package classifier;

import model.Sample;

public class MahalonobisDistanceCalculator implements DistanceCalculator {

	@Override
	public double calculate(Sample sample, ClassStatisticData classStatisticData) {
		return Common.calculateMahalonobisDistance(sample, classStatisticData);
	}

}
