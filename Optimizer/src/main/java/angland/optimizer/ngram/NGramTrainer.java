package angland.optimizer.ngram;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;

import angland.optimizer.Optimizer;
import angland.optimizer.Range;
import angland.optimizer.nn.RnnCellTemplate;
import angland.optimizer.saver.StringContext;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;
import angland.optimizer.var.scalar.Scalar;

public class NGramTrainer {

  public static void train(ExecutorService es, List<List<Integer>> trainSentences,
      File contextPath, int vocabSize, RnnCellTemplate cellTemplate, int batchSize,
      int saveInterval, double stepDistance, int samples) throws IOException {
    Map<IndexedKey<String>, Double> context = null;
    long tokenCount = 0;
    if (contextPath.exists()) {
      System.out.println("Loading context " + contextPath);
      context = StringContext.loadContext(contextPath);
      System.out.println("Load complete.");
    } else {
      System.out.println("Initializing context " + contextPath);
      Map<IndexedKey<String>, Double> tmpContext = new HashMap<>();
      cellTemplate.getKeys().forEach(k -> tmpContext.put(k, 2 * Math.random() - 1));
      context = tmpContext;
      System.out.println("New context initialized.");
    }
    Map<IndexedKey<String>, Range> variableRanges = new HashMap<>();
    context.keySet().forEach(k -> variableRanges.put(k, new Range(-1, 1)));
    NGramPredictor predictor = new NGramPredictor(vocabSize, cellTemplate, context, false);

    long startTime = System.currentTimeMillis();
    for (int i = 0; i < trainSentences.size() / batchSize; ++i) {
      Scalar<String> cumulativeLoss = Scalar.constant(0);
      List<List<Integer>> batch = new ArrayList<>();
      for (int j = 0; j < batchSize; ++j) {
        List<Integer> sequence = trainSentences.get((int) (trainSentences.size() * Math.random()));
        batch.add(sequence);
        tokenCount += sequence.size();
      }
      Scalar<String> loss = predictor.getBatchLoss(batch, es, samples);
      context = Optimizer.step(loss, context, variableRanges, stepDistance);
      predictor = new NGramPredictor(vocabSize, cellTemplate, context, false);
      cumulativeLoss = cumulativeLoss.plus(loss);
      if (i % saveInterval == 0) {
        KeyedDerivative.printRelativeDist(loss.getGradient());
        double timeTakenSeconds = (System.currentTimeMillis() - startTime) / 1000.0;
        double tokensPerSecond = (tokenCount) / timeTakenSeconds;
        double sequencesPerSecond = ((i + 1) * saveInterval * batchSize) / timeTakenSeconds;
        System.out.println("Batch loss " + cumulativeLoss.value());
        System.out.println("Tokens per second " + tokensPerSecond);
        System.out.println("Sequence per second " + sequencesPerSecond);
        StringContext.saveContext(context, contextPath);
      }
    }


  }
}
