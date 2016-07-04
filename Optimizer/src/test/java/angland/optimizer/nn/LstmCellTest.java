package angland.optimizer.nn;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.Test;

import angland.optimizer.var.IMatrixValue;
import angland.optimizer.var.IndexedKey;

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
    LstmCell lstmCell = new LstmCell("cell", 5, context);
    IMatrixValue<String> inHidden = IMatrixValue.var("hidden", 5, 1, context);
    IMatrixValue<String> inExposed = IMatrixValue.var("exposed", 5, 1, context);

    LstmStateTuple<String> in = new LstmStateTuple<>(inHidden, inExposed);
    LstmStateTuple<String> out = lstmCell.apply(in);
    assertNotNull(out.getExposedState());
    assertNotNull(out.getHiddenState());
  }

  /*
   * @Test public void testEncodedStateRetained() { Map<IndexedKey<String>, Double> context = new
   * HashMap<>(); context.put(IndexedKey.matrixKey("inputHidden", 0, 0), 0.5);
   * context.put(IndexedKey.matrixKey("inputExposed", 0, 0), 0.5);
   * context.put(IndexedKey.matrixKey("replaceHidden", 0, 0), 0.5);
   * context.put(IndexedKey.matrixKey("replaceModify", 0, 0), 0.5);
   * context.put(IndexedKey.matrixKey("w", 0, 0), 0.5); context.put(IndexedKey.matrixKey("b", 0, 0),
   * 0.5); IMatrixValue<String> inputHidden = IMatrixValue.var("inputHidden", 1, 1, context);
   * IMatrixValue<String> inputExposed = IMatrixValue.var("inputExposed", 1, 1, context);
   * IMatrixValue<String> retainHidden = IMatrixValue.var("replaceHidden", 1, 1, context);
   * IMatrixValue<String> replaceModify = IMatrixValue.var("replaceModify", 1, 1, context);
   * FeedForwardLayer<String> select = new FeedForwardLayer<>(1, 1, ScalarValue::sigmoid, "w", "b",
   * context);
   * 
   * // lstmCell apply IMatrixValue<String> hiddenModified =
   * inputHidden.multiplyPointwise(retainHidden).plus(replaceModify);
   * 
   * IMatrixValue<String> selector = select.apply(inputExposed);
   * 
   * IMatrixValue<String> cellOutput = selector.multiplyPointwise(hiddenModified);
   * 
   * 
   * LstmStateTuple<String> stateTuple = new LstmStateTuple<>(hiddenModified, cellOutput);
   * 
   * assertNotNull(stateTuple.getExposedState()); assertNotNull(stateTuple.getHiddenState()); }
   * 
   * @Test public void testRetain3() { IMatrixValue<String> embedding =
   * IMatrixValue.var("embedding", 1, 1); LstmCell cell = new LstmCell("cell", 1);
   * Map<IndexedKey<String>, Double> context = new HashMap<>();
   * context.put(IndexedKey.matrixKey("embedding", 0, 0), .5); cell.getKeys().forEach(k ->
   * context.put(k, .5)); List<Integer> inputInts = new ArrayList<>(); inputInts.add(0);
   * IMatrixValue<String> hiddenState = IMatrixValue.repeat(ScalarExpression.constant(0),
   * cell.getSize(), 1); IMatrixValue<String> lastOutput =
   * IMatrixValue.repeat(ScalarExpression.constant(0), cell.getSize(), 1); for (int i : inputInts) {
   * IMatrixValue<String> selectedCol = embedding.getColumn(ScalarExpression.constant(i));
   * 
   * LstmStateTuple<String> inputState = new LstmStateTuple<String>(hiddenState, selectedCol);
   * LstmStateTuple<String> outputState = cell.apply(inputState); Map<Object, Object> solutionMap =
   * new HashMap<>(); IMatrixValue<String> exposedSolution =
   * outputState.getExposedState().evaluateAndCache(context, solutionMap); IMatrixValue<String>
   * hiddenInputState = (IMatrixValue<String>) solutionMap.get(hiddenState); IMatrixValue<String>
   * hiddenOutputState = (IMatrixValue<String>) solutionMap.get(outputState.getHiddenState());
   * assertNotNull(exposedSolution); assertNotNull(hiddenInputState);
   * assertNotNull(hiddenOutputState);
   * 
   * LstmStateTuple<String> outputState = cell.apply(inputState); LstmStateTupleValue<String>
   * outputStateValue = outputState.evaluate(context);
   * assertNotNull(outputStateValue.getExposedState());
   * assertNotNull(outputStateValue.getHiddenState()); LstmStateTuple<String> outputStateAsConstant
   * = outputStateValue.toConstant(); hiddenState = outputStateAsConstant.getHiddenState();
   * lastOutput = outputStateAsConstant.getExposedState();
   * 
   * }
   * 
   * }
   */
}
