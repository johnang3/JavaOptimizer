package angland.optimizer.nn;

import static org.junit.Assert.assertNotNull;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.Test;

import angland.optimizer.var.Context;
import angland.optimizer.var.ContextTemplate;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.matrix.Matrix;

public class LstmCellTest {


  @Test
  public void testHiddenValueRetained() {
    LstmCellTemplate template = new LstmCellTemplate("cell", 5, 0, false);
    Map<IndexedKey<String>, Double> cMap = new HashMap<>();
    Stream.concat(
        template.getKeys(),
        Stream.concat(IndexedKey.getAllMatrixKeys("hidden", 5, 1).stream(), IndexedKey
            .getAllMatrixKeys("exposed", 5, 1).stream())).forEach(k -> {
      cMap.put(k, Math.random() * 2 - 1);
    });
    Context<String> context = ContextTemplate.simpleContext(cMap);
    RnnCell<String> lstmCell = template.create(context);
    Matrix<String> inHidden = Matrix.var("hidden", 5, 1, context);
    Matrix<String> inExposed = Matrix.var("exposed", 5, 1, context);

    RnnStateTuple<String> in = new RnnStateTuple<>(inHidden, inExposed);
    RnnStateTuple<String> out = lstmCell.apply(in);
    assertNotNull(out.getExposedState());
    assertNotNull(out.getHiddenState());
  }

}
