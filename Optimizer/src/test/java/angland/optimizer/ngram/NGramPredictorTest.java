package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.junit.Test;

import angland.optimizer.nn.LstmCellTemplate;
import angland.optimizer.nn.PeepholeLstmCellTemplate;
import angland.optimizer.nn.RnnCellTemplate;
import angland.optimizer.optimizer.GradientDescentOptimizer;
import angland.optimizer.optimizer.Range;
import angland.optimizer.var.Context;
import angland.optimizer.var.ContextKey;
import angland.optimizer.var.ContextTemplate;
import angland.optimizer.var.scalar.IScalarValue;

public class NGramPredictorTest {


  @Test
  public void testOutputHasNoException() {
    RnnCellTemplate template = new LstmCellTemplate("cell", 10, .005, false);
    NGramPredictor predictor =
        new NGramPredictor(20, template, ContextTemplate.simpleContext(NGramPredictor
            .randomizedContext(20, template)), false);
    List<Integer> input = new ArrayList<>();
    input.add(0);
    input.add(1);
    predictor.predictNext(input, 5, -1);
  }

  @Test
  public void testLossHasNoExceptions() {
    RnnCellTemplate template = new LstmCellTemplate("cell", 10, .005, false);
    NGramPredictor predictor =
        new NGramPredictor(20, template, ContextTemplate.simpleContext(NGramPredictor
            .randomizedContext(20, template)), false);
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
    ContextTemplate<String> contextTemplate =
        new ContextTemplate<>(NGramPredictor.getKeys(10, template).collect(Collectors.toList()));
    Context<String> context = contextTemplate.randomContext();
    Map<ContextKey<String>, Range> variableRanges = new HashMap<>();
    context.getContextTemplate().getContextKeys()
        .forEach((k) -> variableRanges.put(k, new Range(-1, 1)));
    NGramPredictor predictor = new NGramPredictor(10, template, context, false);
    List<Integer> input = new ArrayList<>();
    input.add(1);
    input.add(1);
    input.add(1);
    input.add(1);
    for (int i = 0; i < 15; ++i) {
      System.out.println(predictor.predictNext(input, 5, -1));
      IScalarValue<String> loss = predictor.getLoss(input, 5);
      System.out.println(loss.value());

      context =
          contextTemplate.createContext(GradientDescentOptimizer.step(loss, context.asMap(),
              variableRanges, .2));
      predictor = new NGramPredictor(10, template, context, false);
    }
  }


}
