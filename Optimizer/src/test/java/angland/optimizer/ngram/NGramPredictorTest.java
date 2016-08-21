package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import angland.optimizer.Optimizer;
import angland.optimizer.Range;
import angland.optimizer.nn.LstmCellTemplate;
import angland.optimizer.nn.PeepholeLstmCellTemplate;
import angland.optimizer.nn.RnnCellTemplate;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.scalar.Scalar;

public class NGramPredictorTest {


  @Test
  public void testOutputHasNoException() {
    RnnCellTemplate template = new LstmCellTemplate("cell", 10, .005, false);
    NGramPredictor predictor =
        new NGramPredictor(20, template, NGramPredictor.randomizedContext(20, template), false);
    List<Integer> input = new ArrayList<>();
    input.add(0);
    input.add(1);
    predictor.predictNext(input, 5, -1);
  }

  @Test
  public void testLossHasNoExceptions() {
    RnnCellTemplate template = new LstmCellTemplate("cell", 10, .005, false);
    NGramPredictor predictor =
        new NGramPredictor(20, template, NGramPredictor.randomizedContext(20, template), false);
    List<Integer> input = new ArrayList<>();
    input.add(1);
    input.add(2);
    input.add(3);
    input.add(4);
    predictor.getLoss(input, 10).value();
  }

  @Test
  public void testLossReduction() {
    RnnCellTemplate template = new PeepholeLstmCellTemplate("cell", 6, .005, false);

    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    Map<IndexedKey<String>, Range> variableRanges = new HashMap<>();
    NGramPredictor.getKeys(10, template).forEach((k) -> {
      variableRanges.put(k, new Range(-1, 1));
      startingPoint.put(k, 2 * Math.random() - 1);
    });
    Map<IndexedKey<String>, Double> context = startingPoint;
    NGramPredictor predictor = new NGramPredictor(10, template, context, false);
    List<Integer> input = new ArrayList<>();
    input.add(1);
    input.add(1);
    input.add(1);
    input.add(1);
    for (int i = 0; i < 15; ++i) {
      System.out.println(predictor.predictNext(input, 5, -1));
      Scalar<String> loss = predictor.getLoss(input, 5);
      System.out.println(loss.value());

      context = Optimizer.step(loss, context, variableRanges, .2);
      predictor = new NGramPredictor(10, template, context, false);
    }
  }


}
