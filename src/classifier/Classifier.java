package classifier;

import java.util.List;

import model.ModelClass;
import model.Sample;
import model.SampleWithClass;

public interface Classifier {

	void train(List<SampleWithClass> trainSamples);
	
	ModelClass classify(Sample sample);
	
	boolean isTrained();
	
}
