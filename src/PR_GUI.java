
import static java.util.Arrays.asList;
import static java.util.stream.Collectors.toList;

import java.awt.event.ActionEvent;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;

import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JTextField;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.apache.commons.math3.util.Combinations;

import Jama.Matrix;
import classifier.Classifier;
import classifier.Common;
import classifier.EuclideanDistanceCalculator;
import classifier.KNearestMeanClassifier;
import classifier.KNearestNeighborClassifier;
import classifier.MahalonobisDistanceCalculator;
import classifier.NearestMeanClassifier;
import classifier.NearestNeighborClassifier;
import model.ModelClass;
import model.Sample;
import model.SampleWithClass;
import validator.BootstrapValidator;
import validator.ClassificationValidator;
import validator.CrossvalidationValidator;
import validator.SimpleValidator;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * PR_GUI.java
 *
 * Created on 2015-03-05, 19:40:56
 */

/**
 *
 * @author krzy
 */
public class PR_GUI extends javax.swing.JFrame {

	private static final long serialVersionUID = 1L;
	
	String InData; // dataset from a text file will be placed here
    int ClassCount=0, FeatureCount=0;
    double[][] F, FNew; // original feature matrix and transformed feature matrix
    int[] ClassLabels, SampleCount;
    String[] ClassNames;
    
    private List<ModelClass> classes;

    /** Creates new form PR_GUI */
    public PR_GUI() {
        initComponents();
        setSize(720,410);
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        rbg_F = new javax.swing.ButtonGroup();
        b_read = new javax.swing.JButton();
        jPanel2 = new javax.swing.JPanel();
        jLabel1 = new javax.swing.JLabel();
        l_dataset_name_l = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        jLabel4 = new javax.swing.JLabel();
        l_dataset_name = new javax.swing.JLabel();
        l_nfeatures = new javax.swing.JLabel();
        jButton2 = new javax.swing.JButton();
        jPanel3 = new javax.swing.JPanel();
        jLabel5 = new javax.swing.JLabel();
        jLabel6 = new javax.swing.JLabel();
        selbox_nfeat = new javax.swing.JComboBox<String>();
        jSeparator1 = new javax.swing.JSeparator();
        f_rb_extr = new javax.swing.JRadioButton();
        f_rb_sel = new javax.swing.JRadioButton();
        b_deriveFS = new javax.swing.JButton();
        jLabel10 = new javax.swing.JLabel();
        f_combo_criterion = new javax.swing.JComboBox<String>();
        f_combo_PCA_LDA = new javax.swing.JComboBox<String>();
        jLabel12 = new javax.swing.JLabel();
        tf_PCA_Energy = new javax.swing.JTextField();
        jLabel14 = new javax.swing.JLabel();
        jLabel15 = new javax.swing.JLabel();
        l_NewDim = new javax.swing.JLabel();
        jPanel4 = new javax.swing.JPanel();
        jLabel8 = new javax.swing.JLabel();
        jLabel9 = new javax.swing.JLabel();
        jLabe20 = new JLabel();
        jLabe21 = new JLabel();
        resultLabel = new JLabel();
        l_bootstrapIterations = new JLabel();
        validationMethodLabel = new javax.swing.JLabel();
        l_simpleValidationTestSizeInPercent = new JLabel();
        jComboBox2 = new javax.swing.JComboBox<String>();
        validationMethodComboBox = new javax.swing.JComboBox<String>();
        b_Execute = new javax.swing.JButton();
        jLabel16 = new javax.swing.JLabel();
        tf_TrainSetSize = new javax.swing.JTextField();
        jLabel17 = new javax.swing.JLabel();
        jPanel5 = new javax.swing.JPanel();
        jLabel2 = new javax.swing.JLabel();
        l_FLD_winner = new javax.swing.JLabel();
        jLabel13 = new javax.swing.JLabel();
        l_FLD_val = new javax.swing.JLabel();
        tf_bootstrapIterations = new JTextField();
        tf_simpleValidationTestSizeInPercent = new JTextField();
        

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        getContentPane().setLayout(null);

        b_read.setText("Read dataset");
        b_read.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                b_readActionPerformed(evt);
            }
        });
        getContentPane().add(b_read);
        b_read.setBounds(20, 10, 130, 25);

        jPanel2.setBackground(new java.awt.Color(204, 255, 255));
        jPanel2.setBorder(javax.swing.BorderFactory.createEtchedBorder());

        jLabel1.setFont(new java.awt.Font("Comic Sans MS", 0, 18)); // NOI18N
        jLabel1.setText("Dataset info");

        l_dataset_name_l.setText("Name:");

        jLabel3.setText("Classes:");

        jLabel4.setText("Features:");

        l_dataset_name.setText("...");

        l_nfeatures.setText("...");

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel2Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addComponent(l_dataset_name_l)
                        .addGap(18, 18, 18)
                        .addComponent(l_dataset_name))
                    .addComponent(jLabel1))
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addGap(115, 115, 115)
                        .addComponent(jLabel3))
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addGap(94, 94, 94)
                        .addComponent(jLabel4)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(l_nfeatures)))
                .addGap(100, 100, 100))
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel1)
                    .addComponent(jLabel3))
                .addGap(10, 10, 10)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(l_dataset_name_l)
                    .addComponent(jLabel4)
                    .addComponent(l_dataset_name)
                    .addComponent(l_nfeatures))
                .addContainerGap(24, Short.MAX_VALUE))
        );

        getContentPane().add(jPanel2);
        jPanel2.setBounds(10, 50, 320, 80);

        jButton2.setText("Parse dataset");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });
        getContentPane().add(jButton2);
        jButton2.setBounds(190, 10, 130, 25);

        jPanel3.setBackground(new java.awt.Color(255, 255, 204));
        jPanel3.setBorder(javax.swing.BorderFactory.createEtchedBorder());
        jPanel3.setLayout(null);

        jLabel5.setFont(new java.awt.Font("Comic Sans MS", 0, 18)); // NOI18N
        jLabel5.setText("Feature space");
        jPanel3.add(jLabel5);
        jLabel5.setBounds(14, 2, 118, 26);

        jLabel6.setText("FS Dimension");
        jPanel3.add(jLabel6);
        jLabel6.setBounds(178, 9, 78, 16);

        selbox_nfeat.setModel(new javax.swing.DefaultComboBoxModel<String>(new String[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" }));
        selbox_nfeat.setEnabled(true);
        jPanel3.add(selbox_nfeat);
        selbox_nfeat.setBounds(268, 6, 40, 22);
        jPanel3.add(jSeparator1);
        jSeparator1.setBounds(14, 41, 290, 10);

        f_rb_extr.setBackground(new java.awt.Color(255, 255, 204));
        rbg_F.add(f_rb_extr);
        f_rb_extr.setText("Feature extraction");
        f_rb_extr.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                f_rb_extrActionPerformed(evt);
            }
        });
        jPanel3.add(f_rb_extr);
        f_rb_extr.setBounds(10, 110, 133, 25);

        f_rb_sel.setBackground(new java.awt.Color(255, 255, 204));
        rbg_F.add(f_rb_sel);
        f_rb_sel.setSelected(true);
        f_rb_sel.setText("Feature selection");
        f_rb_sel.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                f_rb_selActionPerformed(evt);
            }
        });
        jPanel3.add(f_rb_sel);
        f_rb_sel.setBounds(10, 60, 127, 25);

        b_deriveFS.setText("Derive Feature Space");
        b_deriveFS.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                b_deriveFSActionPerformed(evt);
            }
        });
        jPanel3.add(b_deriveFS);
        b_deriveFS.setBounds(10, 180, 292, 25);

        jLabel10.setText("Criterion");
        jPanel3.add(jLabel10);
        jLabel10.setBounds(200, 50, 49, 16);

        f_combo_criterion.setModel(new javax.swing.DefaultComboBoxModel<String>(new String[] { "Fisher discriminant", "Classification error", "SFS" }));
        f_combo_criterion.setEnabled(true);
        jPanel3.add(f_combo_criterion);
        f_combo_criterion.setBounds(160, 70, 140, 22);

        f_combo_PCA_LDA.setModel(new javax.swing.DefaultComboBoxModel<String>(new String[] { "PCA", "LDA" }));
        f_combo_PCA_LDA.setEnabled(false);
        jPanel3.add(f_combo_PCA_LDA);
        f_combo_PCA_LDA.setBounds(190, 110, 70, 22);

        jLabel12.setText("Energy");
        jPanel3.add(jLabel12);
        jLabel12.setBounds(20, 150, 39, 16);

        tf_PCA_Energy.setText("80");
        jPanel3.add(tf_PCA_Energy);
        tf_PCA_Energy.setBounds(70, 150, 30, 22);

        jLabel14.setText("%");
        jPanel3.add(jLabel14);
        jLabel14.setBounds(110, 150, 20, 16);

        jLabel15.setText("New dimension:");
        jPanel3.add(jLabel15);
        jLabel15.setBounds(160, 150, 92, 16);

        l_NewDim.setText("...");
        jPanel3.add(l_NewDim);
        l_NewDim.setBounds(270, 150, 30, 16);

        getContentPane().add(jPanel3);
        jPanel3.setBounds(10, 140, 320, 220);

        jPanel4.setBackground(new java.awt.Color(204, 255, 204));
        jPanel4.setBorder(javax.swing.BorderFactory.createEtchedBorder());
        jPanel4.setLayout(null);

        jLabel8.setFont(new java.awt.Font("Comic Sans MS", 0, 18)); // NOI18N
        jLabel8.setText("Classifier");
        jPanel4.add(jLabel8);
        jLabel8.setBounds(10, 0, 79, 26);

        jLabel9.setText("Classification method");
        jPanel4.add(jLabel9);
        jLabel9.setBounds(14, 44, 152, 16);

        jComboBox2.setModel(new javax.swing.DefaultComboBoxModel<String>(new String[] { 
        		"Nearest neighbor (NN)", 
        		"Nearest Mean (NM)", 
        		"k-Nearest Neighbor (k-NN)", 
        		"k-NM with Euclidean distance", 
        		"k-NM with Mahalonobis distance" }));
        jPanel4.add(jComboBox2);
        jComboBox2.setBounds(160, 41, 178, 22);
        
        validationMethodLabel.setText("Validation method");
        jPanel4.add(validationMethodLabel);
        validationMethodLabel.setBounds(14, 84, 152, 16);
        
        validationMethodComboBox.setModel(new javax.swing.DefaultComboBoxModel<String>(new String[] { 
        		"Simple validation",
        		"Crossvalidation", 
        		"Bootstrap" }));
        jPanel4.add(validationMethodComboBox);
        validationMethodComboBox.setBounds(160, 81, 178, 22);
        validationMethodComboBox.addActionListener(event -> onValidationMethodChange(event));

        l_simpleValidationTestSizeInPercent.setText("Training set percent size");
        jPanel4.add(l_simpleValidationTestSizeInPercent);
        l_simpleValidationTestSizeInPercent.setBounds(14, 124, 140, 16);
        
        tf_simpleValidationTestSizeInPercent.setText("80");
        jPanel4.add(tf_simpleValidationTestSizeInPercent);
        tf_simpleValidationTestSizeInPercent.setBounds(160, 124, 20, 22);
        
        jLabe21.setText("%");
        jPanel4.add(jLabe21);
        jLabe21.setBounds(185, 124, 20, 22);
        
        jLabel16.setText("Number of sets");
        jLabel16.setVisible(false);
        jPanel4.add(jLabel16);
        jLabel16.setBounds(14, 124, 140, 16);

        tf_TrainSetSize.setText("5");
        tf_TrainSetSize.setVisible(false);
        jPanel4.add(tf_TrainSetSize);
        tf_TrainSetSize.setBounds(160, 124, 20, 22);

        l_bootstrapIterations.setText("Number of iterations");
        jPanel4.add(l_bootstrapIterations);
        l_bootstrapIterations.setBounds(14, 124, 140, 16);
        l_bootstrapIterations.setVisible(false);
        
        tf_bootstrapIterations.setText("50");
        jPanel4.add(tf_bootstrapIterations);
        tf_bootstrapIterations.setBounds(160, 124, 30, 22);
        tf_bootstrapIterations.setVisible(false);
        
        b_Execute.setText("Execute");
        jPanel4.add(b_Execute);
        b_Execute.setBounds(14, 165, 120, 30);
        b_Execute.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                b_ExecuteActionPerformed(evt);;
            }
        });
        
        jLabe20.setText("Result:");
        jPanel4.add(jLabe20);
        jLabe20.setBounds(160, 170, 50, 16);
        
        resultLabel.setText("-");
        jPanel4.add(resultLabel);
        resultLabel.setBounds(210, 170, 80, 16);

        getContentPane().add(jPanel4);
        jPanel4.setBounds(340, 150, 350, 210);

        jPanel5.setBorder(javax.swing.BorderFactory.createTitledBorder("Results"));
        jPanel5.setLayout(null);

        jLabel2.setText("FS Winners:");
        jPanel5.add(jLabel2);
        jLabel2.setBounds(10, 30, 70, 16);

        l_FLD_winner.setText("-");
        jPanel5.add(l_FLD_winner);
        l_FLD_winner.setBounds(100, 30, 240, 16);

        jLabel13.setText("FLD value: ");
        jPanel5.add(jLabel13);
        jLabel13.setBounds(10, 60, 70, 16);

        l_FLD_val.setText("-");
        jPanel5.add(l_FLD_val);
        l_FLD_val.setBounds(100, 60, 240, 16);

        getContentPane().add(jPanel5);
        jPanel5.setBounds(340, 10, 350, 130);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void onValidationMethodChange(ActionEvent event) {
    	if (validationMethodComboBox.getSelectedIndex() == 0) {
    		tf_simpleValidationTestSizeInPercent.setVisible(true);
    		l_simpleValidationTestSizeInPercent.setVisible(true);
    		jLabe21.setVisible(true);
    		
    		jLabel16.setVisible(false);
    		tf_TrainSetSize.setVisible(false);
    		jLabel17.setVisible(false);
    		
    		l_bootstrapIterations.setVisible(false);
    		tf_bootstrapIterations.setVisible(false);
    	} else if (validationMethodComboBox.getSelectedIndex() == 1) {
    		tf_simpleValidationTestSizeInPercent.setVisible(false);
    		l_simpleValidationTestSizeInPercent.setVisible(false);
    		jLabe21.setVisible(false);
    		
    		jLabel16.setVisible(true);
    		tf_TrainSetSize.setVisible(true);
    		jLabel17.setVisible(true);
    		
    		l_bootstrapIterations.setVisible(false);
    		tf_bootstrapIterations.setVisible(false);
    	} else {
    		tf_simpleValidationTestSizeInPercent.setVisible(false);
    		l_simpleValidationTestSizeInPercent.setVisible(false);
    		jLabe21.setVisible(false);
    		
    		jLabel16.setVisible(false);
    		tf_TrainSetSize.setVisible(false);
    		jLabel17.setVisible(false);
    		
    		l_bootstrapIterations.setVisible(true);
    		tf_bootstrapIterations.setVisible(true);
    	}
	}

	private void f_rb_selActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_f_rb_selActionPerformed
        f_combo_criterion.setEnabled(true);
        f_combo_PCA_LDA.setEnabled(false);
        selbox_nfeat.setEnabled(true);
    }//GEN-LAST:event_f_rb_selActionPerformed

    private void f_rb_extrActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_f_rb_extrActionPerformed
        f_combo_criterion.setEnabled(false);
        f_combo_PCA_LDA.setEnabled(true);
        selbox_nfeat.setEnabled(false);
    }//GEN-LAST:event_f_rb_extrActionPerformed

    private void b_readActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_b_readActionPerformed
        // reads in a text file; contents is placed into a variable of String type
        InData = readDataSet();
    }//GEN-LAST:event_b_readActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        // Analyze text inputted from a file: determine class number and labels and number
        // of features; build feature matrix: columns - samples, rows - features
        try {
            if(InData!=null) {
                getDatasetParameters();
                l_nfeatures.setText(FeatureCount+"");
                fillFeatureMatrix();
            }
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this,ex.getMessage());
        }
        
    }//GEN-LAST:event_jButton2ActionPerformed

    private void b_deriveFSActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_b_deriveFSActionPerformed
        // derive optimal feature space
        if (F == null) {
        	return;
        }
        
        if (featureSelectionOptionIsChosen()) {
            int numberOfFeaturesToSelect = Integer.parseInt((String)selbox_nfeat.getSelectedItem());
			List<Integer> bestFeatureIndexes = selectBestFeatureIndexes(numberOfFeaturesToSelect);
            
            classes = new ArrayList<>();
            for (String className : ClassNames) {
    			classes.add(new ModelClass(className));
    		}
            
            for (int j = 0; j < F[0].length; j++) {
            	List<Double> features = new ArrayList<>();
            	
            	for (int featureIndex : bestFeatureIndexes) {
            		features.add(F[featureIndex][j]);
            	}
            	
            	classes
            		.get(ClassLabels[j])
            		.addSample(new Sample(features));
            }
        } else if (featureExtractionOptionIsChosen()) {
            double TotEnergy=Double.parseDouble(tf_PCA_Energy.getText())/100.0;
            // Target dimension (if k>0) or flag for energy-based dimension (k=0)
            int k=0;
//            double[][] FF = { {1,1}, {1,2}};
//            double[][] FF = { {-2,0,2}, {-1,0,1}};
            // F is an array of initial features, FNew is the resulting array
            double[][] FFNorm = centerAroundMean(F); 
            Matrix Cov = computeCovarianceMatrix(FFNorm);
            Matrix TransformMat = extractFeatures(Cov,TotEnergy, k);     
            FNew = projectSamples(new Matrix(FFNorm),TransformMat);
            // FNew is a matrix with samples projected to a new feature space
            l_NewDim.setText(FNew.length+"");
            
            classes = new ArrayList<>();
            for (String className : ClassNames) {
    			classes.add(new ModelClass(className));
    		}
            
            for (int j = 0; j < FNew[0].length; j++) {
            	List<Double> features = new ArrayList<>();
            	
            	for (int i = 0; i < FNew.length; i++) {
            		features.add(FNew[i][j]);
            	}
            	
            	classes
            		.get(ClassLabels[j])
            		.addSample(new Sample(features));
            }
        }
    }//GEN-LAST:event_b_deriveFSActionPerformed

	private boolean featureExtractionOptionIsChosen() {
		return f_rb_extr.isSelected();
	}

	private boolean featureSelectionOptionIsChosen() {
		return f_rb_sel.isSelected();
	}

	private boolean featureSpaceHasNotBeenReduced() {
		return classes == null || classes.isEmpty();
	}
    
	private Classifier instantiateClassifierBasedOnUserSelection() {
		switch (jComboBox2.getSelectedIndex()) {
			case 0:
				return new NearestNeighborClassifier();
			case 1:
				return new NearestMeanClassifier();
			case 2:
				return new KNearestNeighborClassifier();
			case 3:
				return new KNearestMeanClassifier(new EuclideanDistanceCalculator());
			case 4:
				return new KNearestMeanClassifier(new MahalonobisDistanceCalculator());
			default:
				throw new IllegalStateException("Invalid index <" + jComboBox2.getSelectedIndex() + ">");
		}
	}
	
	private void b_ExecuteActionPerformed(java.awt.event.ActionEvent evt) {
		
		if (featureSpaceHasNotBeenReduced()) {
        	return;
        }
        
		Classifier classifier = instantiateClassifierBasedOnUserSelection();
		ClassificationValidator validator = instantiateValidatorBasedOnUserSelection();
		List<SampleWithClass> samples = fetchAllSamples();
        
		double result = validator.validate(classifier, samples) * 100;
		
		resultLabel.setText(new DecimalFormat("#.#").format(result) + " %");
	}
	
	private ClassificationValidator instantiateValidatorBasedOnUserSelection() {
		switch (validationMethodComboBox.getSelectedIndex()) {
			case 0:
				int trainingSetSizeInPercents = Integer.parseInt(tf_simpleValidationTestSizeInPercent.getText());
				return new SimpleValidator(trainingSetSizeInPercents);
			case 1:
				int numberOfSets = Integer.parseInt(tf_TrainSetSize.getText());
				return new CrossvalidationValidator(numberOfSets);
			case 2:
				int numberOfIterations = Integer.parseInt(tf_bootstrapIterations.getText());
				return new BootstrapValidator(numberOfIterations);
			default:
				throw new IllegalStateException("Unsupported index <" + validationMethodComboBox.getSelectedIndex() + ">");
		}
	}
	
	private List<SampleWithClass> fetchAllSamples() {
		List<SampleWithClass> allSamples = new ArrayList<>();
    	classes.stream()
    		.map(toSamplesWithClass)
    		.forEach(samples -> allSamples.addAll(samples));
    	
		return allSamples;
	}
	
	private static Function<ModelClass, List<SampleWithClass>> toSamplesWithClass = new Function<ModelClass, List<SampleWithClass>>() {
		@Override
		public List<SampleWithClass> apply(ModelClass modelClass) {
			return modelClass
					.getSamples()
					.stream()
					.map(sample -> new SampleWithClass(sample, modelClass))
					.collect(toList());
		}
	};

    /**
    * @param args the command line arguments
    */
    public static void main(String args[]) {
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new PR_GUI().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton b_deriveFS;
    private javax.swing.JButton b_read;
    private javax.swing.JComboBox<String> f_combo_PCA_LDA;
    private javax.swing.JComboBox<String> f_combo_criterion;
    private javax.swing.JRadioButton f_rb_extr;
    private javax.swing.JRadioButton f_rb_sel;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton b_Execute;
    private javax.swing.JComboBox<String> jComboBox2;
    private javax.swing.JComboBox<String> validationMethodComboBox;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel10;
    private javax.swing.JLabel jLabel12;
    private javax.swing.JLabel jLabel13;
    private javax.swing.JLabel jLabel14;
    private javax.swing.JLabel jLabel15;
    private javax.swing.JLabel jLabel16;
    private javax.swing.JLabel jLabel17;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel6;
    private javax.swing.JLabel jLabel8;
    private javax.swing.JLabel jLabel9; 
    private javax.swing.JLabel jLabe20;
    private javax.swing.JLabel jLabe21; 
    private javax.swing.JLabel resultLabel; 
    private javax.swing.JLabel validationMethodLabel;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JPanel jPanel4;
    private javax.swing.JPanel jPanel5;
    private javax.swing.JSeparator jSeparator1;
    private javax.swing.JLabel l_FLD_val;
    private javax.swing.JLabel l_FLD_winner;
    private javax.swing.JLabel l_NewDim;
    private javax.swing.JLabel l_dataset_name;
    private javax.swing.JLabel l_dataset_name_l;
    private javax.swing.JLabel l_nfeatures;
    private javax.swing.JLabel l_bootstrapIterations;
    private javax.swing.JLabel l_simpleValidationTestSizeInPercent;
    private javax.swing.ButtonGroup rbg_F;
    private javax.swing.JComboBox<String> selbox_nfeat;
    private javax.swing.JTextField tf_PCA_Energy;
    private javax.swing.JTextField tf_TrainSetSize;
    private javax.swing.JTextField tf_bootstrapIterations;
    private javax.swing.JTextField tf_simpleValidationTestSizeInPercent;
    // End of variables declaration//GEN-END:variables

    private String readDataSet() {

        String s_tmp, s_out="";
        JFileChooser jfc = new JFileChooser();
        jfc.setCurrentDirectory(new File(".."));
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
                                            "Datasets - plain text files", "txt");
        jfc.setFileFilter(filter);
        if(jfc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                BufferedReader br = new BufferedReader(new FileReader(jfc.getSelectedFile()));
                while((s_tmp=br.readLine())!=null) s_out += s_tmp + '$';
                br.close();
                l_dataset_name.setText(jfc.getSelectedFile().getName());
            } catch (Exception e) {        }
        }
        return s_out;
    }

    private void getDatasetParameters() throws Exception{
        // based on data stored in InData determine: class count and names, number of samples 
        // and number of features; set the corresponding variables
        String stmp=InData, saux="";
        // analyze the first line and get feature count: assume that number of features
        // equals number of commas
        saux = InData.substring(InData.indexOf(',')+1, InData.indexOf('$'));
        if(saux.length()==0) throw new Exception("The first line is empty");
        // saux stores the first line beginning from the first comma
        int count=0;
        while(saux.indexOf(',') >0){
            saux = saux.substring(saux.indexOf(',')+1);            
            count++;
        }
        FeatureCount = count+1; // the first parameter
        // Determine number of classes, class names and number of samples per class
        boolean New;
        int index=-1;
        List<String> NameList = new ArrayList<String>();
        List<Integer> CountList = new ArrayList<Integer>();
        List<Integer> LabelList = new ArrayList<Integer>();
        while(stmp.length()>1){
            saux = stmp.substring(0,stmp.indexOf(' '));
            New = true; 
            index++; // new class index
            for(int i=0; i<NameList.size();i++) 
                if(saux.equals(NameList.get(i))) {
                    New=false;
                    index = i; // class index
                }
            if(New) {
                NameList.add(saux);
                CountList.add(0);
            }
            else{
                CountList.set(index, CountList.get(index).intValue()+1);
            }           
            LabelList.add(index); // class index for current row
            stmp = stmp.substring(stmp.indexOf('$')+1);
        }
        // based on results of the above analysis, create variables
        ClassNames = new String[NameList.size()];
        for(int i=0; i<ClassNames.length; i++)
            ClassNames[i]=NameList.get(i);
        SampleCount = new int[CountList.size()];
        for(int i=0; i<SampleCount.length; i++)
            SampleCount[i] = CountList.get(i).intValue()+1;
        ClassLabels = new int[LabelList.size()];
        for(int i=0; i<ClassLabels.length; i++)
            ClassLabels[i] = LabelList.get(i).intValue();
    }

    private void fillFeatureMatrix() throws Exception {
        // having determined array size and class labels, fills in the feature matrix
        int n = 0;
        String saux, stmp = InData;
        for(int i=0; i<SampleCount.length; i++)
            n += SampleCount[i];
        if(n<=0) throw new Exception("no samples found");
        F = new double[FeatureCount][n]; // samples are placed column-wise
        for(int j=0; j<n; j++){
            saux = stmp.substring(0,stmp.indexOf('$'));
            saux = saux.substring(stmp.indexOf(',')+1);
            for(int i=0; i<FeatureCount-1; i++) {
                F[i][j] = Double.parseDouble(saux.substring(0,saux.indexOf(',')));
                saux = saux.substring(saux.indexOf(',')+1);
            }
            F[FeatureCount-1][j] = Double.parseDouble(saux);
            stmp = stmp.substring(stmp.indexOf('$')+1);
        }
        
    }

    private List<Integer> selectBestFeatureIndexes(int numberOfFeaturesToSelect) {
    	if (useFisherLinearDiscriminantOptionIsSelected()) {
    		return selectBestFeatureIndexesUsingFisherLinearDiscriminant(numberOfFeaturesToSelect);
    	} else if (useSFSOptionIsSelected()) {
    		return selectBestFeatureIndexesUsingSFS(numberOfFeaturesToSelect);
    	} else {
    		throw new IllegalArgumentException("Selected method is not implemented");
    	}
    }

	private boolean useFisherLinearDiscriminantOptionIsSelected() {
		return f_combo_criterion.getSelectedIndex() == 0;
	}

	private List<Integer> selectBestFeatureIndexesUsingFisherLinearDiscriminant(int numberOfFeaturesToSelect) {
		if (numberOfFeaturesToSelect == 1) {
            double FLD=0, tmp;
            int bestFeatureIndex = -1;        
            for(int i=0; i<FeatureCount; i++){
                if((tmp=computeFisherLD(F[i]))>FLD){
                    FLD=tmp;
                    bestFeatureIndex = i;
                }
            }
            l_FLD_winner.setText(bestFeatureIndex+"");
            l_FLD_val.setText(FLD+"");
            
            return asList(bestFeatureIndex);
        } else if (numberOfFeaturesToSelect > 1) {
        	int[] bestFeatureIndexes = null;
        	double fisherDiscriminant = Double.MIN_VALUE;
        	
        	Combinations combinations = new Combinations(F.length, numberOfFeaturesToSelect);
        	for (int[] combination : combinations) {
        		double tmp = computeFisherLD(F, combination);
				if (tmp > fisherDiscriminant) {
					fisherDiscriminant = tmp;
					bestFeatureIndexes = combination;
				}
			}
        	
        	List<Integer> listOfBestFeatureIndexes = IntStream.of(bestFeatureIndexes)
				.boxed()
				.collect(toList());
        	
        	l_FLD_winner.setText(listOfBestFeatureIndexes+"");
            l_FLD_val.setText(fisherDiscriminant+"");
        	
        	return listOfBestFeatureIndexes;
        } else {
        	throw new IllegalArgumentException("Illegal number of features <" + numberOfFeaturesToSelect + "> to select");
        }
	}

    private double computeFisherLD(double[] vec) {
        // 1D, 2-classes
        double mA=0, mB=0, sA=0, sB=0;
        for(int i=0; i<vec.length; i++){
            if(ClassLabels[i]==0) {
                mA += vec[i];
                sA += vec[i]*vec[i];
            }
            else {
                mB += vec[i];
                sB += vec[i]*vec[i];
            }
        }
        mA /= SampleCount[0];
        mB /= SampleCount[1];
        sA = sA/SampleCount[0] - mA*mA;
        sB = sB/SampleCount[1] - mB*mB;
        return Math.abs(mA-mB)/(Math.sqrt(sA)+Math.sqrt(sB));
    }
    
    private double computeFisherLD(double[][] featureMatrix, int[] featureIndexes) {
    	List<Sample> samplesOfFirstClass = new ArrayList<>();
    	List<Sample> samplesOfSecondClass = new ArrayList<>();
    	
    	for (int i = 0; i < F[0].length; i++) {
    		List<Double> features = new ArrayList<>();
    		for (int featureIndex : featureIndexes) {
    			features.add(F[featureIndex][i]);
			}
    		
    		if (ClassLabels[i] == 0) {
    			samplesOfFirstClass.add(new Sample(features));
    		} else {
    			samplesOfSecondClass.add(new Sample(features));
    		}
    	}
    	
		Matrix meanOfFirstClass = Common.calculateMean(samplesOfFirstClass);
		Matrix meanOfSecondClass = Common.calculateMean(samplesOfSecondClass);
		
		Matrix covarianceMatrixOfFirstClass = Common.calculateCovarianceMatrix(samplesOfFirstClass, meanOfFirstClass);
		Matrix covarianceMatrixOfSecondClass = Common.calculateCovarianceMatrix(samplesOfSecondClass, meanOfSecondClass);
		
		return Common.calculateEuclideanDistance(meanOfFirstClass, meanOfSecondClass) / (covarianceMatrixOfFirstClass.det() + covarianceMatrixOfSecondClass.det());
    }

    private boolean useSFSOptionIsSelected() {
		return f_combo_criterion.getSelectedIndex() == 2;
	}
    
    private List<Integer> selectBestFeatureIndexesUsingSFS(int numberOfFeaturesToSelect) {
    	List<Integer> bestFeatureIndexes = new ArrayList<>(numberOfFeaturesToSelect);
    	
    	double FLD=0, tmp;
        int bestFeatureIndex = -1;        
        for(int i=0; i<FeatureCount; i++){
            if((tmp=computeFisherLD(F[i]))>FLD){
                FLD=tmp;
                bestFeatureIndex = i;
            }
        }
        bestFeatureIndexes.add(bestFeatureIndex);
        
		for (int i = 1; i < numberOfFeaturesToSelect; i++) {
			double fisherDiscriminant = Double.MIN_VALUE;
			bestFeatureIndexes.add(-1);
			
			for (int j = 0; j < F.length; j++) {
				if (bestFeatureIndexes.contains(j)) {
					continue;
				}
				
				int[] featureIndexes = new int[i+1];
				for (int k = 0; k < i; k++) {
					featureIndexes[k] = bestFeatureIndexes.get(k);
				}
				featureIndexes[i] = j;
				
        		tmp = computeFisherLD(F, featureIndexes);
				if (tmp > fisherDiscriminant) {
					fisherDiscriminant = tmp;
					bestFeatureIndexes.set(i, j);
				}
			}
		}
		
    	l_FLD_winner.setText(bestFeatureIndexes+"");
        l_FLD_val.setText("-");
		
		return bestFeatureIndexes;
	}
    
	private Matrix extractFeatures(Matrix C, double Ek, int k) {               
        
        Matrix evecs, evals;
        // compute eigenvalues and eigenvectors
        evecs = C.eig().getV();
        evals = C.eig().getD();
        
        // PM: projection matrix that will hold a set dominant eigenvectors
        Matrix PM;
        if(k>0) {
            // preset dimension of new feature space
//            PM = new double[evecs.getRowDimension()][k];
            PM = evecs.getMatrix(0, evecs.getRowDimension()-1, 
                    evecs.getColumnDimension()-k, evecs.getColumnDimension()-1);
        }
        else {
            // dimension will be determined based on scatter energy
            double TotEVal = evals.trace(); // total energy
            double EAccum=0;
            int m=evals.getColumnDimension()-1;
            while(EAccum<Ek*TotEVal){
                EAccum += evals.get(m, m);
                m--;
            }
            PM = evecs.getMatrix(0, evecs.getRowDimension()-1,m+1,evecs.getColumnDimension()-1);
        }

/*            System.out.println("Eigenvectors");                
            for(int i=0; i<r; i++){
                for(int j=0; j<c; j++){
                    System.out.print(evecs[i][j]+" ");
                }
                System.out.println();                
            }
            System.out.println("Eigenvalues");                
            for(int i=0; i<r; i++){
                for(int j=0; j<c; j++){
                    System.out.print(evals[i][j]+" ");
                }
                System.out.println();                
            }
*/
        
        return PM;
    }

    private Matrix computeCovarianceMatrix(double[][] m) {
//        double[][] C = new double[M.length][M.length];
        
        Matrix M = new Matrix(m);
        Matrix MT = M.transpose();       
        Matrix C = M.times(MT);
        return C;
    }

    private double[][] centerAroundMean(double[][] M) {
        
        double[] mean = new double[M.length];
        for(int i=0; i<M.length; i++)
            for(int j=0; j<M[0].length; j++)
                mean[i]+=M[i][j];
        for(int  i=0; i<M.length; i++) mean[i]/=M[0].length;
        for(int i=0; i<M.length; i++)
            for(int j=0; j<M[0].length; j++)
                M[i][j]-=mean[i];
        return M;
    }

    private double[][] projectSamples(Matrix FOld, Matrix TransformMat) {
        
        return (FOld.transpose().times(TransformMat)).transpose().getArrayCopy();
    }
}