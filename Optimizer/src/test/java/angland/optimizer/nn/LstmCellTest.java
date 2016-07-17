package angland.optimizer.nn;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.Test;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.matrix.IMatrixValue;

public class LstmCellTest {

  @Test
  public void testStreamKeys() {
    assertEquals(90, LstmCell.getKeys("cell", 5).count());
  }

  @Test
  public void testHiddenValueRetained() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    Stream.concat(
        LstmCell.getKeys("cell", 5),
        Stream.concat(IndexedKey.getAllMatrixKeys("hidden", 5, 1).stream(), IndexedKey
            .getAllMatrixKeys("exposed", 5, 1).stream())).forEach(k -> {
      context.put(k, Math.random() * 2 - 1);
    });
    LstmCell lstmCell = new LstmCell("cell", 5, context, 0, false);
    IMatrixValue<String> inHidden = IMatrixValue.var("hidden", 5, 1, context);
    IMatrixValue<String> inExposed = IMatrixValue.var("exposed", 5, 1, context);

    LstmStateTuple<String> in = new LstmStateTuple<>(inHidden, inExposed);
    LstmStateTuple<String> out = lstmCell.apply(in);
    assertNotNull(out.getExposedState());
    assertNotNull(out.getHiddenState());
  }

}
