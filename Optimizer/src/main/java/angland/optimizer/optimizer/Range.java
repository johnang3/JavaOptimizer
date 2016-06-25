package angland.optimizer.optimizer;

public class Range {

  private final double max;
  private final double min;

  public Range(double max, double min) {
    super();
    if (max <= min) {
      throw new IllegalArgumentException("Max must be greater than min.");
    }
    this.max = max;
    this.min = min;
  }

  public double getMax() {
    return max;
  }

  public double getMin() {
    return min;
  }



}
