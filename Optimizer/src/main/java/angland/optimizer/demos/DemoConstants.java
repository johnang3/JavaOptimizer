package angland.optimizer.demos;

import angland.optimizer.nn.PeepholeLstmCellTemplate;
import angland.optimizer.nn.RnnCellTemplate;

class DemoConstants {
  static int vocabSize = 10000;
  static int lstmSize = 30;
  static double gradientClipThreshold = .2;

  public static RnnCellTemplate getTemplate(boolean constant) {
    return new PeepholeLstmCellTemplate("cell", lstmSize, gradientClipThreshold, constant);

    /*
     * return FeatureGroupRnnCellTemplate.simple("cell", lstmSize, 4, gradientClipThreshold,
     * constant, true);
     */
  }
}
