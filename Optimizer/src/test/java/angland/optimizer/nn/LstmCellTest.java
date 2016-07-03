package angland.optimizer.nn;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.Test;

import angland.optimizer.var.IMatrixValue;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.MatrixExpression;
import angland.optimizer.var.ScalarExpression;
import angland.optimizer.var.ScalarValue;

public class LstmCellTest {

  @Test
  public void testStreamKeys() {
    LstmCell lstmCell = new LstmCell("cell", 5);
    assertEquals(90, lstmCell.getKeys().count());
  }

  @Test
  public void testHiddenValueRetained() {
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
    LstmStateTupleValue<String> value = out.evaluate(context);
    assertNotNull(value.getExposedState());
    assertNotNull(value.getHiddenState());
  }

  @Test
  public void testEncodedStateRetained() {
    MatrixExpression<String> inputHidden = MatrixExpression.variable("inputHidden", 1, 1);
    MatrixExpression<String> inputExposed = MatrixExpression.variable("inputExposed", 1, 1);
    MatrixExpression<String> retainHidden = MatrixExpression.variable("replaceHidden", 1, 1);
    MatrixExpression<String> replaceModify = MatrixExpression.variable("replaceModify", 1, 1);
    FeedForwardLayer<String> select = new FeedForwardLayer<>(1, 1, ScalarValue::sigmoid, "w", "b");

    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.matrixKey("inputHidden", 0, 0), 0.5);
    context.put(IndexedKey.matrixKey("inputExposed", 0, 0), 0.5);
    context.put(IndexedKey.matrixKey("replaceHidden", 0, 0), 0.5);
    context.put(IndexedKey.matrixKey("replaceModify", 0, 0), 0.5);
    context.put(IndexedKey.matrixKey("w", 0, 0), 0.5);
    context.put(IndexedKey.matrixKey("b", 0, 0), 0.5);

    // lstmCell apply
    MatrixExpression<String> hiddenModified =
        inputHidden.pointwiseMultiply(retainHidden).plus(replaceModify);

    MatrixExpression<String> selector = select.apply(inputExposed);

    MatrixExpression<String> cellOutput = selector.pointwiseMultiply(hiddenModified);

    // evaluate
    Map<Object, Object> partialSolutions = new HashMap<>();
    IMatrixValue<String> cellOutputSolution = cellOutput.evaluate(context, partialSolutions);
    @SuppressWarnings("unchecked")
    IMatrixValue<String> hiddenModifiedSolution =
        (IMatrixValue<String>) partialSolutions.get(hiddenModified);
    assertNotNull(cellOutputSolution);
    assertNotNull(hiddenModifiedSolution);

    LstmStateTupleExpression<String> stateTupleExpression =
        new LstmStateTupleExpression<>(hiddenModified, cellOutput);
    LstmStateTupleValue<String> stateTupleValue = stateTupleExpression.evaluate(context);
    assertNotNull(stateTupleValue.getExposedState());
    assertNotNull(stateTupleValue.getHiddenState());
  }

  @Test
  public void testRetain3() {
    MatrixExpression<String> embedding = MatrixExpression.variable("embedding", 1, 1);
    LstmCell cell = new LstmCell("cell", 1);
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.matrixKey("embedding", 0, 0), .5);
    cell.getKeys().forEach(k -> context.put(k, .5));
    List<Integer> inputInts = new ArrayList<>();
    inputInts.add(0);
    MatrixExpression<String> hiddenState =
        MatrixExpression.repeat(ScalarExpression.constant(0), cell.getSize(), 1);
    MatrixExpression<String> lastOutput =
        MatrixExpression.repeat(ScalarExpression.constant(0), cell.getSize(), 1);
    for (int i : inputInts) {
      MatrixExpression<String> selectedCol = embedding.getColumn(ScalarExpression.constant(i));

      LstmStateTupleExpression<String> inputState =
          new LstmStateTupleExpression<String>(hiddenState, selectedCol);
      LstmStateTupleExpression<String> outputState = cell.apply(inputState);
      Map<Object, Object> solutionMap = new HashMap<>();
      IMatrixValue<String> exposedSolution =
          outputState.getExposedState().evaluateAndCache(context, solutionMap);
      IMatrixValue<String> hiddenInputState = (IMatrixValue<String>) solutionMap.get(hiddenState);
      IMatrixValue<String> hiddenOutputState =
          (IMatrixValue<String>) solutionMap.get(outputState.getHiddenState());
      assertNotNull(exposedSolution);
      assertNotNull(hiddenInputState);
      assertNotNull(hiddenOutputState);
      /*
       * LstmStateTupleExpression<String> outputState = cell.apply(inputState);
       * LstmStateTupleValue<String> outputStateValue = outputState.evaluate(context);
       * assertNotNull(outputStateValue.getExposedState());
       * assertNotNull(outputStateValue.getHiddenState()); LstmStateTupleExpression<String>
       * outputStateAsConstant = outputStateValue.toConstant(); hiddenState =
       * outputStateAsConstant.getHiddenState(); lastOutput =
       * outputStateAsConstant.getExposedState();
       */
    }

  }
}
