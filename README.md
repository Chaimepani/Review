# Review
A machine learning-based Fake Review Detector that analyzes text patterns and sentiment to identify whether a product/service review is genuine or fake. Built using Python, NLP, and classification models
<dependencies>
    <dependency>
        <groupId>nz.ac.waikato.cms.weka</groupId>
        <artifactId>weka-stable</artifactId>
        <version>3.8.6</version>
    </dependency>
    <dependency>
        <groupId>com.opencsv</groupId>
        <artifactId>opencsv</artifactId>
        <version>5.7.1</version>
    </dependency>
</dependencies>
import java.util.*;
import java.util.regex.Pattern;

public class TextPreprocessor {
    private static final Set<String> STOP_WORDS = Set.of("the", "is", "at", "which", "on", "and", "a", "to", "as", "i", "it", "for", "of", "an", "in"); // Add more stopwords
    private static final Pattern PUNCTUATION = Pattern.compile("[^a-zA-Z\\s]");

    public static String preprocess(String text) {
        if (text == null) return "";
        // Lowercase
        text = text.toLowerCase();
        // Remove punctuation
        text = PUNCTUATION.matcher(text).replaceAll(" ");
        // Tokenize and remove stopwords (simple split)
        String[] tokens = text.split("\\s+");
        List<String> cleanedTokens = new ArrayList<>();
        for (String token : tokens) {
            if (!STOP_WORDS.contains(token) && token.length() > 2) { // Basic stemming: remove common suffixes (e.g., "ing", "ed")
                token = stemSimple(token);
                cleanedTokens.add(token);
            }
        }
        return String.join(" ", cleanedTokens);
    }

    private static String stemSimple(String word) {
        return word.replaceAll("ing$", "").replaceAll("ed$", "").replaceAll("s$", ""); // Very basic; use SnowballStemmer for better
    }
}
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class FakeReviewDetector {
    private NaiveBayes model;
    private StringToWordVector filter;

    public void train(String csvFilePath) throws Exception {
        // Load CSV data
        Instances data = loadCSV(csvFilePath);
        if (data == null || data.numInstances() == 0) {
            throw new IllegalArgumentException("No data loaded");
        }

        // Preprocess text (apply to 'REVIEW_TEXT' attribute)
        preprocessData(data);

        // Convert to ARFF (Weka format) and apply TF-IDF
        filter = new StringToWordVector();
        filter.setIDFTransform(true); // TF-IDF
        filter.setLowerCaseTokens(true);
        filter.setWordsToKeep(5000); // Max features
        filter.setNGrams(1, 2); // Unigrams + bigrams
        data = Filter.useFilter(data, filter);

        // Set class index
        data.setClassIndex(data.numAttributes() - 1);

        // Train model
        model = new NaiveBayes();
        model.buildClassifier(data);

        // Save ARFF for reuse (optional)
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("processed.arff"));
        saver.writeBatch();

        System.out.println("Model trained successfully!");
    }

    private Instances loadCSV(String filePath) throws IOException, CsvValidationException {
        List<String[]> records = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            String[] headers = reader.readNext(); // Skip header
            String[] record;
            while ((record = reader.readNext()) != null) {
                records.add(record);
            }
        }

        // Create Instances
        FastVector attributes = new FastVector();
        attributes.addElement(new Attribute("REVIEW_TEXT", (FastVector) null));
        FastVector classValues = new FastVector();
        classValues.addElement("genuine"); // 0
        classValues.addElement("fake");    // 1
        attributes.addElement(new Attribute("LABEL", classValues));

        Instances data = new Instances("reviews", attributes, records.size());
        for (String[] row : records) {
            if (row.length >= 2) {
                Instance inst = new DenseInstance(2);
                inst.setValue(0, row[0]); // REVIEW_TEXT
                inst.setValue(1, row[1].equals("1") ? "fake" : "genuine"); // LABEL
                data.add(inst);
            }
        }
        return data;
    }

    private void preprocessData(Instances data) {
        for (int i = 0; i < data.numInstances(); i++) {
            String originalText = data.instance(i).stringValue(0);
            String cleaned = TextPreprocessor.preprocess(originalText);
            data.instance(i).setValue(0, cleaned);
        }
    }

    public String predict(String reviewText) throws Exception {
        if (model == null || filter == null) {
            throw new IllegalStateException("Model not trained");
        }
        String cleaned = TextPreprocessor.preprocess(reviewText);
        Instances testData = new Instances(filter.getOutputFormat(), 0);
        testData.add(new DenseInstance(1.0));
        testData.instance(0).setValue(0, cleaned);
        testData.setClassIndex(0); // Temp for filtering
        testData = Filter.useFilter(testData, filter);
        testData.setClassIndex(testData.numAttributes() - 1);

        double pred = model.classifyInstance(testData.instance(0));
        return pred == 1.0 ? "Fake" : "Genuine";
    }

    // Getter for model (for evaluation)
    public NaiveBayes getModel() { return model; }
    public StringToWordVector getFilter() { return filter; }
}
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class EvaluateModel {
    public static void evaluate(FakeReviewDetector detector, Instances testData) throws Exception {
        Evaluation eval = new Evaluation(testData);
        eval.evaluateModel(detector.getModel(), testData);
        System.out.println("Accuracy: " + String.format("%.2f", eval.pctCorrect() / 100));
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());
    }

    // Usage: Split data manually or use Weka's split
    public static void main(String[] args) throws Exception {
        FakeReviewDetector detector = new FakeReviewDetector();
        detector.train("yelp.csv"); // Train on full data; in practice, split 80/20

        // Example prediction
        System.out.println("Prediction: " + detector.predict("This product is amazing! I love it so much, it's the best ever!!!"));

        // For evaluation, load test data and call evaluate(detector, testData);
    }
}
