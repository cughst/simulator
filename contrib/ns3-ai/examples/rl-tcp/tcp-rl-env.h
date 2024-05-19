/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Technische Universität Berlin
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz@tkn.tu-berlin.de>
 * Modify: Pengyu Liu <eic_lpy@hust.edu.cn> 
 *         Hao Yin <haoyin@uw.edu>
 */

#pragma once
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/ns3-ai-module.h"
namespace ns3 {

// tcp rl environment struct
//基于结构体
struct sTcpRlEnv
{
  uint32_t nodeId; 
  uint32_t socketUid;
  uint8_t envType;
  int64_t simTime_us;//仿真时间

  uint32_t ssThresh; //action  state
  uint32_t cWnd;  //action  state
  
  uint32_t segmentSize;//state
  uint32_t segmentsAcked;//state
  uint32_t bytesInFlight;//state

} Packed;

//action
struct TcpRlAct
{
  uint32_t new_ssThresh;
  uint32_t new_cWnd;
};

// 继承环境
class TcpRlEnv : public Ns3AIRL<sTcpRlEnv, TcpRlAct>
{
public:
  TcpRlEnv () = delete;
  TcpRlEnv (uint16_t id);
  void SetNodeId (uint32_t id); //获取节点id
  void SetSocketUuid (uint32_t id);     //获取socket的id
  // 数据包轨迹跟踪
  void TxPktTrace (Ptr<const Packet>, const TcpHeader &, Ptr<const TcpSocketBase>);
  void RxPktTrace (Ptr<const Packet>, const TcpHeader &, Ptr<const TcpSocketBase>);
  
  //get ssthresh cwind
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) = 0;
  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) = 0;

  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time &rtt) = 0;
  virtual void CongestionStateSet (Ptr<TcpSocketState> tcb,
                                   const TcpSocketState::TcpCongState_t newState) = 0;
  virtual void CwndEvent (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) = 0;

protected:
  uint32_t m_nodeId;
  uint32_t m_socketUuid;
  //游戏是否结束  奖励
  bool m_isGameOver;
  float m_envReward;
  
  Time m_lastPktTxTime{MicroSeconds (0.0)};
  Time m_lastPktRxTime{MicroSeconds (0.0)};
  uint64_t m_interTxTimeNum{0};
  Time m_interTxTimeSum{MicroSeconds (0.0)};
  uint64_t m_interRxTimeNum{0};
  Time m_interRxTimeSum{MicroSeconds (0.0)};
  
  //存储action的结果
  uint32_t m_new_ssThresh;
  uint32_t m_new_cWnd;
};

//创建时间步的环境
class TcpTimeStepEnv : public TcpRlEnv
{
public:
  TcpTimeStepEnv () = delete;
  TcpTimeStepEnv (uint16_t id);

  // TCP congestion control interface
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight);
  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked);
  // optional functions used to collect obs
  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time &rtt);
  virtual void CongestionStateSet (Ptr<TcpSocketState> tcb,
                                   const TcpSocketState::TcpCongState_t newState);
  virtual void CwndEvent (Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event);

private:
  void ScheduleNextStateRead ();
  bool m_started{false};
  Time m_timeStep{MilliSeconds(10)};
  // state
  Ptr<const TcpSocketState> m_tcb;
  //vector
  std::vector<uint32_t> m_bytesInFlight;
  std::vector<uint32_t> m_segmentsAcked;

  uint64_t m_rttSampleNum{0};
  Time m_rttSum{MicroSeconds (0.0)};
};

} // namespace ns3
