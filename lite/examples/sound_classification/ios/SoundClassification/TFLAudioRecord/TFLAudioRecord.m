// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "TFLAudioRecord.h"
#import "TFLAudioError.h"
#import "TFLRingBuffer.h"
#import "TFLUtils.h"

#define SUPPORTED_CHANNEL_COUNT 1

@implementation TFLAudioRecord {
  AVAudioEngine *_audioEngine;

  /* Specifying a custom buffer size on AVAUdioEngine while tapping does not take effect. Hence we
   * are storing the returned samples in a ring buffer to acheive the desired buffer size. If
   * specified buffer size is shorter than the buffer size supported by AVAUdioEngine only the most
   * recent data of the buffer of size bufferSize will be stored by the ring buffer. */
  TFLRingBuffer *_ringBuffer;
  dispatch_queue_t _conversionQueue;
  BOOL _tapStateSuccess;
}

- (nullable instancetype)initWithAudioFormat:(TFLAudioFormat *)audioFormat
                                 sampleCount:(NSUInteger)sampleCount
                                       error:(NSError *_Nullable *)error {
  self = [self init];
  if (self) {
    if (audioFormat.channelCount > SUPPORTED_CHANNEL_COUNT) {
      [TFLUtils createCustomError:error
                         withCode:TFLAudioErrorCodeInvalidArgumentError
                      description:@"The channel count provided does not match the supported "
                                  @"channel count. Only 1 audio channel is currently supported."];
      return nil;
    }
    _audioFormat = audioFormat;
    _audioEngine = [[AVAudioEngine alloc] init];
    _bufferSize = sampleCount * audioFormat.channelCount;
    _ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:sampleCount * audioFormat.channelCount];
    _conversionQueue = dispatch_queue_create("com.tflAudio.AudioConversionQueue", NULL);
    _tapStateSuccess = YES;
  }
  return self;
}

/**
 * Uses AVAudioConverter to acheive the desired sample rate since the tap on the input node raises
 * an exception if any format other than the input nodes default format is specified. completion
 * handler structure is used as opposed to polling since this is the pattern followed by all iOS
 * APIs delivering continuous data on a background thread. Even if we implement a read method, it
 * will have to return data in a callback since installTapOnBus delivers data in a callback on a
 * different thread. There will also be extra overhead to ensure thread safety to make sure that
 * reads and writes happen on the same thread sine TFLAudioTensor buffer is meant to be non local.
 */
- (BOOL)startTappingMicrophoneWithError:(NSError **)error {
  //    (void (^)(TFLFloatBuffer *_Nullable buffer, NSError *_Nullable error))completionHandler {
  AVAudioNode *inputNode = [_audioEngine inputNode];
  AVAudioFormat *format = [inputNode outputFormatForBus:0];

  AVAudioFormat *recordingFormat =
      [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                       sampleRate:format.sampleRate
                                         channels:(AVAudioChannelCount)self.audioFormat.channelCount
                                      interleaved:YES];

  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:format
                                                                     toFormat:recordingFormat];

  // Setting buffer size takes no effect on the input node. This class uses a ring buffer internally
  // to ensure the requested buffer size.
  [inputNode
      installTapOnBus:0
           bufferSize:(AVAudioFrameCount)self.bufferSize
               format:recordingFormat
                block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
                  dispatch_async(self->_conversionQueue, ^{
                    // Capacity of converted PCM buffer is calculated in order to maintain the same
                    // latency as the input pcmBuffer.
                    AVAudioFrameCount capacity =
                        ceil(buffer.frameLength * recordingFormat.sampleRate / format.sampleRate);

                    AVAudioPCMBuffer *pcmBuffer = [[AVAudioPCMBuffer alloc]
                        initWithPCMFormat:recordingFormat
                            frameCapacity:capacity *
                                          (AVAudioFrameCount)self.audioFormat.channelCount];

                    AVAudioConverterInputBlock inputBlock =
                        ^AVAudioBuffer *_Nullable(AVAudioPacketCount inNumberOfPackets,
                                                  AVAudioConverterInputStatus *_Nonnull outStatus) {
                      *outStatus = AVAudioConverterInputStatus_HaveData;
                      return buffer;
                    };

                    NSError *conversionError = nil;
                    AVAudioConverterOutputStatus converterStatus =
                        [audioConverter convertToBuffer:pcmBuffer
                                                  error:&conversionError
                                     withInputFromBlock:inputBlock];

                    switch (converterStatus) {
                      case AVAudioConverterOutputStatus_HaveData: {
                            
                        if (pcmBuffer.frameLength == 0) {
                          self->_tapStateSuccess = NO;
                        } else if ((pcmBuffer.frameLength % recordingFormat.channelCount) != 0) {
                          self->_tapStateSuccess = NO;
                        } else if (![self->_ringBuffer loadBuffer:[[TFLFloatBuffer alloc] initWithData:pcmBuffer.floatChannelData[0]
                                                                                                  size:pcmBuffer.frameLength]
                                                           offset:0
                                                             size:pcmBuffer.frameLength
                                                            error:nil]) {
                          self->_tapStateSuccess = NO;

                        } else {
                          NSLog(@"%f",pcmBuffer.floatChannelData[0][0]);
                          self->_tapStateSuccess = YES;
                        }
                        break;
                      }
                      case AVAudioConverterOutputStatus_Error: {
                        self->_tapStateSuccess = NO;
                        break;
                      }
                      default:
                        self->_tapStateSuccess = NO;
                        break;
                    }
                  });
                }];

  [_audioEngine prepare];
  [_audioEngine startAndReturnError:error];
  if (error) {
    return NO;
  }
  
  return YES;
}

- (void)startRecordingWithCompletionHandler:(void (^)(NSError *_Nullable))completionHandler {
  //    (void (^)(TFLFloatBuffer *_Nullable buffer, NSError *_Nullable error))completionHandler {
  [[AVAudioSession sharedInstance] requestRecordPermission:^(BOOL granted) {
    if (granted) {
      NSError *tapError = nil;
      [self startTappingMicrophoneWithError:&tapError];
      completionHandler(tapError);
    } else {
      NSError *permissionError = nil;
      [TFLUtils createCustomError:&permissionError
                         withCode:TFLAudioErrorCodeRecordPermissionDeniedError
                      description:@"User denied the permission to record mic input."];

      completionHandler(permissionError);
    }
  }];
}

- (nullable TFLFloatBuffer *)readAtOffset:(NSUInteger)offset
                        withSize:(NSUInteger)size
                           error:(NSError *_Nullable *)error {
  if (!_tapStateSuccess) {
    [TFLUtils createCustomError:error
                       withCode:TFLAudioErrorCodeAudioProcessingError
                    description:@"Some error occured during audio processing"];
    return nil;
  }
  if (offset + size > _ringBuffer.buffer.size) {
    [TFLUtils createCustomError:error
                       withCode:TFLAudioErrorCodeInvalidArgumentError
                    description:@"Index out of bounds: offset + size should be <= to the size of "
                                @"TFLAudioRecord's internal buffer. "];
    return nil;
  }
  __block TFLFloatBuffer *bufferToReturn;
  bufferToReturn = [[TFLFloatBuffer alloc] initWithSize:size];
  dispatch_sync(_conversionQueue, ^{
    memcpy(bufferToReturn.data, _ringBuffer.buffer.data + offset, size);
  });

  return bufferToReturn;
}

- (void)stop {
  [[_audioEngine inputNode] removeTapOnBus:0];
  [_audioEngine stop];
}

@end
