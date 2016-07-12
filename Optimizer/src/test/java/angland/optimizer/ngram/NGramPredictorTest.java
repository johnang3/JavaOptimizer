package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import angland.optimizer.optimizer.GradientDescentOptimizer;
import angland.optimizer.optimizer.Range;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.scalar.IScalarValue;

public class NGramPredictorTest {


  @Test
  public void testOutputHasNoException() {
    NGramPredictor predictor =
        new NGramPredictor(20, 10, NGramPredictor.randomizedContext(20, 10), .0005);
    List<Integer> input = new ArrayList<>();
    input.add(0);
    input.add(1);
    predictor.predictNext(input, 5);
  }

  @Test
  public void testLossHasNoExceptions() {
    NGramPredictor predictor =
        new NGramPredictor(20, 10, NGramPredictor.randomizedContext(20, 10), .0005);
    List<Integer> input = new ArrayList<>();
    input.add(1);
    input.add(2);
    input.add(3);
    input.add(4);
    predictor.getLoss(input, 10).value();
  }

  @Test
  public void testLossReduction() {
    Map<IndexedKey<String>, Double> context = NGramPredictor.randomizedContext(8, 6);
    Map<IndexedKey<String>, Range> variableRanges = new HashMap<>();
    context.forEach((k, v) -> variableRanges.put(k, new Range(-1, 1)));
    NGramPredictor predictor = new NGramPredictor(8, 6, context, .0005);
    List<Integer> input = new ArrayList<>();
    input.add(1);
    input.add(1);
    input.add(1);
    input.add(1);
    for (int i = 0; i < 50; ++i) {
      System.out.println(predictor.predictNext(input, 5));
      IScalarValue<String> loss = predictor.getLoss(input, 5);
      System.out.println(loss.value());
      context = GradientDescentOptimizer.step(loss, context, variableRanges, 1);
      predictor = new NGramPredictor(8, 6, context, .0005);
    }
  }


}
