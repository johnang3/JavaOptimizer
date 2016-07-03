package angland.optimizer.ngram;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public class NGramPredictorTest {


  @Test
  public void test() {
    NGramPredictor predictor = new NGramPredictor(10, 10);
    List<Integer> input = new ArrayList<>();
    input.add(0);
    input.add(1);
    System.out.println(predictor.predictNext(input, 5));
  }

}
