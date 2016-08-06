package angland.optimizer.demos;

import angland.optimizer.nn.FeatureGroupRnnCellTemplate;
import angland.optimizer.nn.RnnCellTemplate;

class DemoConstants {
  static int vocabSize = 10000;
  static int lstmSize = 120;
  static double gradientClipThreshold = .1;

  public static RnnCellTemplate getTemplate(boolean constant) {
    /*
     * return new StoredPotentialRnnCellTemplate("cell", lstmSize, gradientClipThreshold, constant);
     */

    return FeatureGroupRnnCellTemplate.simple("cell", lstmSize, 4, gradientClipThreshold, constant,
        true);


    // return new PeepholeLstmCellTemplate("cell", lstmSize, gradientClipThreshold, constant);
  }
}
