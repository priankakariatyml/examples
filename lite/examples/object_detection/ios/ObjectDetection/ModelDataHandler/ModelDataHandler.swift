// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLiteTaskVision
import UIKit
import Accelerate

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let inferences: [Inference]
}

/// Stores one formatted inference.
struct Inference {
  let confidence: Float
  let className: String
  let rect: CGRect
  let displayColor: UIColor
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet SSD model.
enum MobileNetSSD {
  static let modelInfo: FileInfo = (name: "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29", extension: "tflite")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {

  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int

  let threshold: Float = 0.5
  
  let inputWidth = 300
  let inputHeight = 300
  
  let threadCountLimit = 10

  // MARK: Private properties

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var objectDetector: ObjectDetector

  private let colorStrideValue = 10
  private let colors = [
    UIColor.red,
    UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
    UIColor.green,
    UIColor.orange,
    UIColor.blue,
    UIColor.purple,
    UIColor.magenta,
    UIColor.yellow,
    UIColor.cyan,
    UIColor.brown
  ]

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    let options = ObjectDetectorOptions(modelPath: modelPath)
    options.baseOptions.computeSettings.cpuSettings.numThreads = Int32(threadCount)
    options.classificationOptions.maxResults = 3
    
    do {
      // Create the `Interpreter`.
      objectDetector = try ObjectDetector.objectDetector(options: options)
     
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    super.init()

  }

  /// This class handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
  /// results for a successful inference.
  func runModel(on sampleBuffer: CMSampleBuffer) -> Result? {

    do {
      guard let gmlImage = MLImage(sampleBuffer: sampleBuffer) else {
        return nil;
      }
      let objectDetectionResult = try objectDetector.detect(gmlImage: gmlImage)
      
      print("Top 3 Objects Detected")

      for (idx, detection) in objectDetectionResult.detections.enumerated() {
        guard let label = detection.categories[0].label else {
          return nil;
        }
        print("\(idx + 1): Label: \(label), Score: \(detection.categories[0].score) Bbox: \(detection.boundingBox)")
      }
      
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

   return nil
  }

}
