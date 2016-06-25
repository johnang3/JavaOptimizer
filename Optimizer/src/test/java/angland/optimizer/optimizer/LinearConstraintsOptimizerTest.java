package angland.optimizer.optimizer;

import static angland.optimizer.var.Expression.constant;
import static angland.optimizer.var.Expression.var;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import angland.optimizer.var.Expression;
import angland.optimizer.vec.OrientedPlane;

public class LinearConstraintsOptimizerTest {

  private static final double TOLERANCE = 10e-3;

  @Test
  public void testNoConstraints() {
    Expression<String> aSquared = var("a").power(2);
    Expression<String> bSquared = var("b").power(2);
    Expression<String> objectiveFunction = aSquared.plus(bSquared);
    Map<String, Double> startingPoint = new HashMap<>();
    startingPoint.put("a", 200.0);
    startingPoint.put("b", 200.0);
    /*
     * Calculation<String> solution = BasicLinearConstraintsOptimizer.optimize(objectiveFunction,
     * new ArrayList<>(), startingPoint, 10000, 10e-6); assertEquals(0.0, solution.value(),
     * TOLERANCE); assertEquals(0.0, solution.getContext().get("a"), TOLERANCE); assertEquals(0.0,
     * solution.getContext().get("b"), TOLERANCE);
     */
  }

  /**
   * Minimize <br/>
   * x^3 * y^2 <br/>
   * <br/>
   * Subject to: <br/>
   * x >= 0; y >= 0; <br/>
   * -x - 2y + 10 >= 0. <br/>
   * <br/>
   * Derivation of solution: <br/>
   * <br/>
   * Minimize U = x^3 y^2 + lambda(x+2y-10); <br/>
   * dU/dx = 3 x^2 y^2 + lambda dU/dy = 2x^3 * y + lambda <br/>
   * du/dy = 2 x^3 y + 2*lambda <br/>
   * lambda = -3 x^2 y^2 = - x^3 y <br/>
   * 3y = x <br/>
   * y = 2 <br/>
   * x = 6 <br/>
   */
  @Test
  public void testCobDouglas2Variables() {
    Expression<String> xCubed = var("x").power(3);
    Expression<String> ySquared = var("y").power(2);
    Expression<String> objective = xCubed.times(ySquared).times(constant(-1));
    OrientedPlane<String> xConstraint = OrientedPlane.minimum("x", 0.0);
    OrientedPlane<String> yConstraint = OrientedPlane.minimum("y", 0.0);
    Map<String, Double> costNormal = new HashMap<>();
    costNormal.put("x", -1.0);
    costNormal.put("y", -2.0);
    OrientedPlane<String> costConstraint = new OrientedPlane<>(costNormal, +10);
    List<OrientedPlane<String>> constraints = new ArrayList<>();
    constraints.add(xConstraint);
    constraints.add(yConstraint);
    constraints.add(costConstraint);
    Map<String, Double> startingPoint = new HashMap<>();
    startingPoint.put("x", 1.0);
    startingPoint.put("y", 1.0);
    /*
     * Calculation<String> solution = BasicLinearConstraintsOptimizer.optimize(objective,
     * constraints, startingPoint, 1, 10e-3); assertEquals(2, solution.getContext().get("y"),
     * TOLERANCE); assertEquals(6, solution.getContext().get("x"), TOLERANCE); assertEquals(-864.0,
     * solution.value(), TOLERANCE);
     */
  }

  /**
   * Verify that we can solve a linear program. <br/>
   * Minimize: <br/>
   * - 8x - 12y <br/>
   * Subject to: <br/>
   * x >= 0 <br/>
   * y >= 0 <br/>
   * -10x - 20y + 140 >= 0 <br/>
   * -6x - 8y + 72 >= 0 <br/>
   * The optimizer should find that the vertex x=8, y=3 is optimal.
   */
  @Test
  public void testLinearProgram() {
    Expression<String> objective = var("x").times(constant(-8)).plus(var("y").times(constant(-12)));
    OrientedPlane<String> xConstraint = OrientedPlane.minimum("x", 0);
    OrientedPlane<String> yConstraint = OrientedPlane.minimum("y", 0);
    Map<String, Double> constraintMap1 = new HashMap<>();
    constraintMap1.put("x", -10.0);
    constraintMap1.put("y", -20.0);
    OrientedPlane<String> constraint1 = new OrientedPlane<>(constraintMap1, 140);
    Map<String, Double> constraintMap2 = new HashMap<>();
    constraintMap2.put("x", -6.0);
    constraintMap2.put("y", -8.0);
    OrientedPlane<String> constraint2 = new OrientedPlane<>(constraintMap2, 72);
    List<OrientedPlane<String>> constraints = new ArrayList<>();
    constraints.add(xConstraint);
    constraints.add(yConstraint);
    constraints.add(constraint1);
    constraints.add(constraint2);
    Map<String, Double> startingPoint = new HashMap<>();
    startingPoint.put("x", 0.0);
    startingPoint.put("y", 0.0);
    /*
     * Calculation<String> solution = BasicLinearConstraintsOptimizer.optimize(objective,
     * constraints, startingPoint, 1, 10e-5); assertEquals(8.0, solution.getContext().get("x"),
     * TOLERANCE); assertEquals(6.0, solution.getContext().get("y"), TOLERANCE); assertEquals(100.0,
     * solution.value(), TOLERANCE);
     */
  }
}
