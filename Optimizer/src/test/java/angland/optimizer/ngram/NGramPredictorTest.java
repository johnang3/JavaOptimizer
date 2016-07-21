package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.junit.Test;

import angland.optimizer.optimizer.GradientDescentOptimizer;
import angland.optimizer.optimizer.Range;
import angland.optimizer.var.Context;
import angland.optimizer.var.ContextKey;
import angland.optimizer.var.ContextTemplate;
import angland.optimizer.var.scalar.IScalarValue;

public class NGramPredictorTest {


  @Test
  public void testOutputHasNoException() {
    NGramPredictor predictor =
        new NGramPredictor(20, 10, ContextTemplate.simpleContext(NGramPredictor.randomizedContext(
            20, 10)), .0005, false);
    List<Integer> input = new ArrayList<>();
    input.add(0);
    input.add(1);
    predictor.predictNext(input, 5);
  }

  @Test
  public void testLossHasNoExceptions() {
    NGramPredictor predictor =
        new NGramPredictor(20, 10, ContextTemplate.simpleContext(NGramPredictor.randomizedContext(
            20, 10)), .0005, false);
    List<Integer> input = new ArrayList<>();
    input.add(1);
    input.add(2);
    input.add(3);
    input.add(4);
    predictor.getLoss(input, 10).value();
  }

  @Test
  public void testLossReduction() {
    ContextTemplate<String> contextTemplate =
        new ContextTemplate<>(NGramPredictor.getKeys(8, 6).collect(Collectors.toList()));
    Context<String> context = contextTemplate.randomContext();
    Map<ContextKey<String>, Range> variableRanges = new HashMap<>();
    context.getContextTemplate().getContextKeys()
        .forEach((k) -> variableRanges.put(k, new Range(-1, 1)));
    NGramPredictor predictor = new NGramPredictor(8, 6, context, .0005, false);
    List<Integer> input = new ArrayList<>();
    input.add(1);
    input.add(1);
    input.add(1);
    input.add(1);
    for (int i = 0; i < 50; ++i) {
      System.out.println(predictor.predictNext(input, 5));
      IScalarValue<String> loss = predictor.getLoss(input, 5);
      System.out.println(loss.value());

      context =
          contextTemplate.createContext(GradientDescentOptimizer.step(loss, context.asMap(),
              variableRanges, .2));
      predictor = new NGramPredictor(8, 6, context, .0005, false);
    }
  }


}
