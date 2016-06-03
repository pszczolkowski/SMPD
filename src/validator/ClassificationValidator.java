package validator;

import java.util.List;

import classifier.Classifier;
import model.SampleWithClass;

public interface ClassificationValidator {

	double validate(Classifier classifier, List<SampleWithClass> samples);
	
}
