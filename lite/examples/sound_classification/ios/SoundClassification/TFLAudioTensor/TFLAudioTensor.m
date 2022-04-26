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

#import "TFLAudioTensor.h"
#import "TFLAudioError.h"
#import "TFLUtils.h"

@implementation TFLAudioTensor {
  TFLRingBuffer *_ringBuffer;
}

- (instancetype)initWithAudioFormat:(TFLAudioFormat *)format sampleCount:(NSUInteger)sampleCount {
  self = [self init];
  if (self) {
    _audioFormat = format;
    _ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:sampleCount * format.channelCount];
  }
  return self;
}

- (BOOL)loadBuffer:(TFLFloatBuffer *)floatBuffer
                offset:(NSInteger)offset
                  size:(NSInteger)size
                 error:(NSError **)error {
  return [_ringBuffer loadBuffer:floatBuffer offset:offset size:floatBuffer.size error:error];
}

- (BOOL)loadAudioRecordBuffer:(TFLFloatBuffer *)floatBuffer withError:(NSError **)error {
  // Sample rate and channel count need not be checked here as they will be checked and reported
  // when TFLAudioRecord which created this record buffer starts emitting wrong sized buffers.

  // Checking buffer size makes sure that channel count and buffer size match.
  if (_ringBuffer.buffer.size != floatBuffer.size) {
    [TFLUtils createCustomError:error
                       withCode:TFLAudioErrorCodeInvalidArgumentError
                    description:@"Size of TFLAudioRecord buffer does not match TFLAudioTensor's buffer "
                                @"size. Please make sure that the TFLAudioRecord object which "
                                @"created floatBuffer is initialized with the same format "
                                @"(channels, sampleRate) and buffer size as TFLAudioTensor."];
    return NO;
  }

  return [self loadBuffer:floatBuffer offset:0 size:floatBuffer.size error:error];
}

- (TFLFloatBuffer *)buffer {
  return [_ringBuffer.buffer copy];
}

@end
