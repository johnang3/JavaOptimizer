package angland.optimizer.nn;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.Test;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.MatrixExpression;

public class LstmCellTest {

  @Test
  public void testStreamKeys() {
    LstmCell lstmCell = new LstmCell("cell", 5);
    assertEquals(90, lstmCell.getKeys().count());
  }

  @Test
  public void testFeedForward() {
    LstmCell lstmCell = new LstmCell("cell", 5);
    MatrixExpression<String> inHidden = MatrixExpression.variable("hidden", 5, 1);
    MatrixExpression<String> inExposed = MatrixExpression.variable("exposed", 5, 1);
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    Stream.concat(
        lstmCell.getKeys(),
        Stream.concat(IndexedKey.getAllMatrixKeys("hidden", 5, 1).stream(), IndexedKey
            .getAllMatrixKeys("exposed", 5, 1).stream())).forEach(k -> {
      context.put(k, Math.random() * 2 - 1);
    });
    LstmStateTupleExpression<String> in = new LstmStateTupleExpression<>(inHidden, inExposed);
    LstmStateTupleExpression<String> out = lstmCell.apply(in);
    out.evaluate(context);
  }
}
