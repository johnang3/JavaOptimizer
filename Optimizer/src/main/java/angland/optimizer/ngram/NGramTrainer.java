package angland.optimizer.ngram;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;

import angland.optimizer.optimizer.GradientDescentOptimizer;
import angland.optimizer.optimizer.Range;
import angland.optimizer.saver.StringContext;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.scalar.IScalarValue;
import angland.optimizer.var.scalar.MappedDerivativeScalar;

public class NGramTrainer {

  public static void train(ExecutorService es, List<List<Integer>> trainSentences,
      File contextPath, int vocabSize, int lstmSize, int batchSize, int saveInterval,
      double stepDistance) throws IOException {
    Map<IndexedKey<String>, Double> context = null;
    if (contextPath.exists()) {
      context = StringContext.loadContext(contextPath);
    } else {
      context = NGramPredictor.randomizedContext(vocabSize, lstmSize);
    }
    Map<IndexedKey<String>, Range> variableRanges = new HashMap<>();
    context.forEach((k, v) -> variableRanges.put(k, new Range(-1, 1)));
    NGramPredictor predictor = new NGramPredictor(vocabSize, lstmSize, context);
    IScalarValue<String> cumulativeLoss = IScalarValue.constant(0);
    long startTime = System.currentTimeMillis();
    for (int i = 0; i < trainSentences.size() / batchSize; ++i) {
      List<List<Integer>> batch = new ArrayList<>();
      for (int j = 0; j < batchSize; ++j) {
        batch.add(trainSentences.get((int) (trainSentences.size() * Math.random())));
      }
      MappedDerivativeScalar<String> loss = predictor.getBatchLoss(batch, es);
      context = GradientDescentOptimizer.step(loss, context, variableRanges, stepDistance);
      predictor = new NGramPredictor(vocabSize, lstmSize, context);
      cumulativeLoss = cumulativeLoss.plus(loss);
      if (i % saveInterval == 0) {
        double timeTakenSeconds = (System.currentTimeMillis() - startTime) / 1000.0;
        double sequencesPerSecond = (saveInterval * batchSize * i) / timeTakenSeconds;
        System.out.println("Batch loss " + cumulativeLoss.value());
        System.out.println("Sequences per second " + sequencesPerSecond);
        StringContext.saveContext(context, contextPath);
        cumulativeLoss = IScalarValue.constant(0);
      }
    }


  }
}
