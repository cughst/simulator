#include "ns3/ipv4.h"
#include "ns3/packet.h"
#include "ns3/ipv4-header.h"
#include "ns3/pause-header.h"
#include "ns3/flow-id-tag.h"
#include "ns3/boolean.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "switch-node.h"
#include "qbb-net-device.h"
#include "ppp-header.h"
#include "ns3/int-header.h"
#include "ns3/simulator.h"
#include <cmath>
#include <iomanip> 
#include <ns3/rdma-hw.h>
#include "ns3/random-variable-stream.h"
#include "ns3/rdma-queue-pair.h"
namespace ns3 {
// extern u_int64_t m_c;
TypeId SwitchNode::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SwitchNode")
    .SetParent<Node> ()
    .AddConstructor<SwitchNode> ()
	.AddAttribute("EcnEnabled",
			"Enable ECN marking.",
			BooleanValue(false),
			MakeBooleanAccessor(&SwitchNode::m_ecnEnabled),
			MakeBooleanChecker())
	.AddAttribute("CcMode",
			"CC mode.",
			UintegerValue(0),
			MakeUintegerAccessor(&SwitchNode::m_ccMode),
			MakeUintegerChecker<uint32_t>())
	.AddAttribute("AckHighPrio",
			"Set high priority for ACK/NACK or not",
			UintegerValue(0),
			MakeUintegerAccessor(&SwitchNode::m_ackHighPrio),
			MakeUintegerChecker<uint32_t>())
	.AddAttribute("MaxRtt",
			"Max Rtt of the network",
			UintegerValue(9000),
			MakeUintegerAccessor(&SwitchNode::m_maxRtt),
			MakeUintegerChecker<uint32_t>())
  ;
  return tid;
}
SwitchNode::SwitchNode()
{
	m_ecmpSeed = GetId();
	m_node_type = 1;
	// buffer
	m_mmu = CreateObject<SwitchMmu>();
	m_rdma = CreateObject<RdmaHw>();
	

	for (uint32_t i = 0; i < pCnt; i++)
		for (uint32_t j = 0; j < pCnt; j++)
			for (uint32_t k = 0; k < qCnt; k++)
			//257*8*8
				m_bytes[i][j][k] = 0;
	for (uint32_t i = 0; i < pCnt; i++)
		//257
		m_txBytes[i] = 0;
	for (uint32_t i = 0; i < pCnt; i++)
		m_lastPktSize[i] = m_lastPktTs[i] = 0;
	for (uint32_t i = 0; i < pCnt; i++)
		m_u[i] = 0;
}

int SwitchNode::GetOutDev(Ptr<const Packet> p, CustomHeader &ch){
	// look up entries
	auto entry = m_rtTable.find(ch.dip);

	// no matching entry
	if (entry == m_rtTable.end())
		return -1;

	// entry found
	auto &nexthops = entry->second;

	// pick one next hop based on hash
	union {
		uint8_t u8[4+4+2+2];
		uint32_t u32[3];
	} buf;
	buf.u32[0] = ch.sip;
	buf.u32[1] = ch.dip;
	if (ch.l3Prot == 0x6)
		buf.u32[2] = ch.tcp.sport | ((uint32_t)ch.tcp.dport << 16);
	else if (ch.l3Prot == 0x11)
		buf.u32[2] = ch.udp.sport | ((uint32_t)ch.udp.dport << 16);
	else if (ch.l3Prot == 0xFC || ch.l3Prot == 0xFD)
		buf.u32[2] = ch.ack.sport | ((uint32_t)ch.ack.dport << 16);

	uint32_t idx = EcmpHash(buf.u8, 12, m_ecmpSeed) % nexthops.size();
	return nexthops[idx];
}

void SwitchNode::CheckAndSendPfc(uint32_t inDev, uint32_t qIndex){
	Ptr<QbbNetDevice> device = DynamicCast<QbbNetDevice>(GetDevice(inDev));
	if (m_mmu->CheckShouldPause(inDev, qIndex)){
		device->SendPfc(qIndex, 0);
		m_mmu->SetPause(inDev, qIndex);
		
	}
}
void SwitchNode::CheckAndSendResume(uint32_t inDev, uint32_t qIndex){
	Ptr<QbbNetDevice> device = DynamicCast<QbbNetDevice>(GetDevice(inDev));
	if (m_mmu->CheckShouldResume(inDev, qIndex)){
		device->SendPfc(qIndex, 1);
		m_mmu->SetResume(inDev, qIndex);
	}
}

void SwitchNode::SendToDev(Ptr<Packet>p, CustomHeader &ch){
	

	int idx = GetOutDev(p, ch);

	// Ptr<RdmaQueuePair> qp = m_rdma->GetQp(ch.sip, ch.cnp.fid, idx);
	// std::cout<<"alpha "<<<<"\n";

	if (idx >= 0){
		Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(GetDevice(idx));

		

		NS_ASSERT_MSG(dev->IsLinkUp(), "The routing table look up should return link that is up");

		// determine the qIndex
		uint32_t qIndex;
		if (ch.l3Prot == 0xFF || ch.l3Prot == 0xFE || (m_ackHighPrio && (ch.l3Prot == 0xFD || ch.l3Prot == 0xFC))){  //QCN or PFC or NACK, go highest priority
			qIndex = 0;
		}else{
			qIndex = (ch.l3Prot == 0x06 ? 1 : ch.udp.pg); // if TCP, put to queue 1
		}
		

		// admission control
		FlowIdTag t;
		p->PeekPacketTag(t);

		uint32_t inDev = t.GetFlowId();
		if (qIndex != 0){ //not highest priority
			if (m_mmu->CheckIngressAdmission(inDev, qIndex, p->GetSize()) && m_mmu->CheckEgressAdmission(idx, qIndex, p->GetSize())){			// Admission control
				m_mmu->UpdateIngressAdmission(inDev, qIndex, p->GetSize());
				m_mmu->UpdateEgressAdmission(idx, qIndex, p->GetSize());
				
			}else{
			}
			CheckAndSendPfc(inDev, qIndex);


		}

		m_bytes[inDev][idx][qIndex] += p->GetSize();
		


		dev->SwitchSend(qIndex, p, ch);
		// for (uint32_t i = 0; i < dev->GetRdmaQueue()->GetFlowCount(); i++){
		// 	Ptr<RdmaQueuePair> qp = dev->GetRdmaQueue()->GetQp(i);
		// 	std::cout<<qp->m_rate<<"\n";}

		// std::cout<<ch.udp.pg<<ch.sip<<ch.dip<<ch.udp.sport<<ch.udp.dport<<"\n";
		// qp = CreateObject<RdmaQueuePair>(u_int16_t(ch.udp.pg),Ipv4Address(ch.dip),Ipv4Address(ch.sip),ch.ack.dport,ch.ack.sport);
		// std::cout<<qp->m_rate<<qp->m_max_rate<<"\n";
		// std::cout<<ch.ack.flags<<"\t"<<"\n";
		//uint64_t t_now = Simulator::Now().GetTimeStep();
		//uint64_t dt = t_now - m_lastPktTs[idx];
		//double txRate = m_lastPktSize[idx] / double(dt)*double(1000000000); // B/ns

		// std::cout<<"in device id is \t  "<<inDev<<"\n";
		// std::cout<<"time last is"<<m_lastPktTs[idx];
		// std::cout<<"time now is \t"<<t_now <<"\n";
		// std::cout<<"next device id is \t"<<idx<<"\n";
		// std::cout <<"tx rate is \t"<< dt<<"\n";
		// std::cout<<"counter of tx bytes at port "<<m_txBytes[idx]<<"\n";
		// float a = m_mmu->EgressBytes(idx,qIndex);
		// std::cout <<"qlen all "<<a<<'\n';
		// std::cout<<"\n";
		
	
	}else
		return; // Drop
	

}

uint32_t SwitchNode::EcmpHash(const uint8_t* key, size_t len, uint32_t seed) {
  uint32_t h = seed;
  if (len > 3) {
    const uint32_t* key_x4 = (const uint32_t*) key;
    size_t i = len >> 2;
    do {
      uint32_t k = *key_x4++;
      k *= 0xcc9e2d51;
      k = (k << 15) | (k >> 17);
      k *= 0x1b873593;
      h ^= k;
      h = (h << 13) | (h >> 19);
      h += (h << 2) + 0xe6546b64;
    } while (--i);
    key = (const uint8_t*) key_x4;
  }
  if (len & 3) {
    size_t i = len & 3;
    uint32_t k = 0;
    key = &key[i - 1];
    do {
      k <<= 8;
      k |= *key--;
    } while (--i);
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
  }
  h ^= len;
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

void SwitchNode::SetEcmpSeed(uint32_t seed){
	m_ecmpSeed = seed;
}

void SwitchNode::AddTableEntry(Ipv4Address &dstAddr, uint32_t intf_idx){
	uint32_t dip = dstAddr.Get();
	m_rtTable[dip].push_back(intf_idx);
}

void SwitchNode::ClearTable(){
	m_rtTable.clear();
}

// This function can only be called in switch mode
bool SwitchNode::SwitchReceiveFromDevice(Ptr<NetDevice> device, Ptr<Packet> packet, CustomHeader &ch){
	SendToDev(packet, ch);
	
	return true;
}

void SwitchNode::SwitchNotifyDequeue(uint32_t ifIndex, uint32_t qIndex, Ptr<Packet> p){
	FlowIdTag t;
	p->PeekPacketTag(t);
	// u_int32_t m_p[257]={0};
	// u_int32_t *ptr = m_lastPktSize;
	
	// u_int32_t array2d_i[257][8]={0};
	// u_int32_t array2d_e[257][8]={0};
	// u_int32_t (*ptr1)[8] = array2d_i;
	// u_int32_t (*ptr2)[8] = array2d_e;

	Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(GetDevice(ifIndex));
	// dev->Receive(p);
	uint32_t inDev = t.GetFlowId();
	if (qIndex != 0){
		
		// 2024 3 5 17:25
		
		// for (u_int32_t i = 0; i < pCnt; i++) {
        // 			for (u_int32_t j = 0; j < qCnt; j++) {
        //    				 array2d_i[i][j] = m_mmu->ingress_bytes[i][j]; // ptr access
        // 		}
		//		}
		int num = 0;
		int sum = 0;
		for (u_int32_t i = 0; i < pCnt; i++) {
			for (u_int32_t j = 0; j < qCnt; j++) {
					// array2d define in switchnode.h flie
					
					array2d[i][j] = m_mmu->egress_bytes[i][j]; // ptr access
					
					if(array2d[i][j] != 0){
						sum +=array2d[i][j];
						num++;	
					}else{
					}
			}
		}
		// get average egress qlen
		qlen_value = int(sum/num);
		
		//qlen_value  = m_mmu->egress_bytes[7][3];

		m_mmu->RemoveFromIngressAdmission(inDev, qIndex, p->GetSize());
		m_mmu->RemoveFromEgressAdmission(ifIndex, qIndex, p->GetSize());
		
		// m_lastPktSize[inDev]=p->GetSize();
		//std::cout<<"device "<<inDev<<"\t"<<"to device "<<ifIndex<<"\n";
		
		/*
		if (inDev != 7)
		for (u_int32_t i = 0; i < pCnt; i++)
		{	
			if (i == inDev && i != 7)
				m_lastPktSize[i] = p->GetSize();
			else
				m_lastPktSize[i] = 0;
		}
		if (ifIndex == 7)
			m_lastPktSize[ifIndex] = p->GetSize();
		// u_int16_t cc = 0; 
		*/

		if (m_ecnEnabled){
			bool egressCongested = m_mmu->ShouldSendCN(ifIndex, qIndex);
			if (egressCongested){
				PppHeader ppp;
				Ipv4Header h;
				p->RemoveHeader(ppp);
				p->RemoveHeader(h);
				h.SetEcn((Ipv4Header::EcnType)0x03);
				p->AddHeader(h);
				p->AddHeader(ppp);
				
				// get marked packet 
				ecn_marked_packet +=p->GetSize();
				
			}
		}
		//CheckAndSendPfc(inDev, qIndex);
		CheckAndSendResume(inDev, qIndex);
	}
	m_txBytes[ifIndex] += p->GetSize();
	if (1){
		uint8_t* buf = p->GetBuffer();
		if (buf[PppHeader::GetStaticSize() + 9] == 0x11){ // udp packet
			IntHeader *ih = (IntHeader*)&buf[PppHeader::GetStaticSize() + 20 + 8 + 6]; // ppp, ip, udp, SeqTs, INT
			Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(GetDevice(ifIndex));
			// std::cout<<dev->GetDataRate().GetBitRate()<<"\n";

			if (m_ccMode == 3){ // HPCC
				ih->PushHop(Simulator::Now().GetTimeStep(), m_txBytes[ifIndex], dev->GetQueue()->GetNBytesTotal(), dev->GetDataRate().GetBitRate());
			}
			else if (m_ccMode == 10){ // HPCC-PINT
				uint64_t t = Simulator::Now().GetTimeStep();
				uint64_t dt = t - m_lastPktTs[ifIndex];
				if (dt > m_maxRtt)
					dt = m_maxRtt;
				uint64_t B = dev->GetDataRate().GetBitRate() / 8; //Bps
				uint64_t qlen = dev->GetQueue()->GetNBytesTotal();
				double newU;

				/**************************
				 * approximate calc
				 *************************/
				int b = 20, m = 16, l = 20; // see log2apprx's paremeters
				int sft = logres_shift(b,l);
				double fct = 1<<sft; // (multiplication factor corresponding to sft)
				double log_T = log2(m_maxRtt)*fct; // log2(T)*fct
				double log_B = log2(B)*fct; // log2(B)*fct
				double log_1e9 = log2(1e9)*fct; // log2(1e9)*fct
				double qterm = 0;
				double byteTerm = 0;
				double uTerm = 0;
				if((qlen >> 8) > 0){
					int log_dt = log2apprx(dt, b, m, l); // ~log2(dt)*fct
					int log_qlen = log2apprx(qlen >> 8, b, m, l); // ~log2(qlen / 256)*fct
					qterm = pow(2, (
								log_dt + log_qlen + log_1e9 - log_B - 2*log_T
								)/fct
							) * 256;
					// 2^((log2(dt)*fct+log2(qlen/256)*fct+log2(1e9)*fct-log2(B)*fct-2*log2(T)*fct)/fct)*256 ~= dt*qlen*1e9/(B*T^2)
				}
				if (m_lastPktSize[ifIndex] > 0){
					int byte = m_lastPktSize[ifIndex];
					int log_byte = log2apprx(byte, b, m, l);
					byteTerm = pow(2, (
								log_byte + log_1e9 - log_B - log_T
								)/fct
							);
					// 2^((log2(byte)*fct+log2(1e9)*fct-log2(B)*fct-log2(T)*fct)/fct) ~= byte*1e9 / (B*T)
				}
				if (m_maxRtt > dt && m_u[ifIndex] > 0){
					int log_T_dt = log2apprx(m_maxRtt - dt, b, m, l); // ~log2(T-dt)*fct
					int log_u = log2apprx(int(round(m_u[ifIndex] * 8192)), b, m, l); // ~log2(u*512)*fct
					uTerm = pow(2, (
								log_T_dt + log_u - log_T
								)/fct
							) / 8192;
					// 2^((log2(T-dt)*fct+log2(u*512)*fct-log2(T)*fct)/fct)/512 = (T-dt)*u/T
				}
				newU = qterm+byteTerm+uTerm;

				#if 0
				/**************************
				 * accurate calc
				 *************************/
				double weight_ewma = double(dt) / m_maxRtt;
				double u;
				if (m_lastPktSize[ifIndex] == 0)
					u = 0;
				else{
					double txRate = m_lastPktSize[ifIndex] / double(dt); // B/ns
					u = (qlen / m_maxRtt + txRate) * 1e9 / B;
				}
				newU = m_u[ifIndex] * (1 - weight_ewma) + u * weight_ewma;
				printf(" %lf\n", newU);
				#endif

				/************************
				 * update PINT header
				 ***********************/
				uint16_t power = Pint::encode_u(newU);
				if (power > ih->GetPower())
					ih->SetPower(power);

				m_u[ifIndex] = newU;
			}else if (m_ccMode == 0)//2024 3 4 add
			{
				uint64_t t = Simulator::Now().GetTimeStep();
				uint64_t dt = t - m_lastPktTs[ifIndex];
				double u;
				
				#if 0
				if (dt > m_maxRtt)
					dt = m_maxRtt;
				#endif

				uint64_t B = dev->GetDataRate().GetBitRate() / 8; //Bps
				//uint64_t qlen = dev->GetQueue()->GetNBytesTotal();

				#if 1
				if (m_lastPktSize[ifIndex] == 0)
					u = 0;
				else{
					double txRate = m_lastPktSize[ifIndex] / double(dt); // B/ns
					u = (txRate) * 1e9 / B;
					tx_rate = u;
				}
				#endif
				/**************************
				 * accurate calc egress port tx utilization
				 *************************/
				#if 0
				double newU;
				uint64_t qlen = dev->GetQueue()->GetNBytesTotal();
				double weight_ewma = double(dt) / m_maxRtt;
				if (m_lastPktSize[ifIndex] == 0)
					u = 0;
				else{
					double txRate = m_lastPktSize[ifIndex] / double(dt); // B/ns
					u = (qlen / m_maxRtt + txRate) * 1e9 / B;
				}
				newU = m_u[ifIndex] * (1 - weight_ewma) + u * weight_ewma;
				tx_rate = newU;
				#endif

				//m_u[ifIndex] = newU;
				///////////////////////
				//sim_ecn.cc get duration 
				if (duration){
					tx_rate_m = ecn_marked_packet*(1e9)/double(duration*B);
					ecn_marked_packet = 0;
				}else{
					tx_rate_m =0;

				}
				min_thresh = m_mmu->kmin[ifIndex];
				max_thresh = m_mmu->kmax[ifIndex];
				pmax_thresh = int((m_mmu->pmax[ifIndex])*100);

				// tx_rate_m = m_c/double(1e11);

				// uint64_t B = dev->GetDataRate().GetBitRate()/8 ; //Bps
				//uint64_t qlen = dev->GetQueue()->GetNBytesTotal();
				//double txRate = (m_lastPktSize[ifIndex] / double(dt))*1000000000;

				// std::cout<<m_mmu->egress_bytes<<"\n";

				// std::cout<<"band width is \t"<<B<<"\n";

				// std::cout<<"time now is \t"<<t<<"\t dt is \t"<<dt<<"\n";
				
				// std::cout<<"port txrate is \t"<< std::fixed << std::setprecision(0)<<txRate<<"\n";
				// // std::cout<<"egress qlen is \t"<<sum<<"\n";
				// // "average is \t"<<sum/total<<"\n";
				// std::cout<<"\n";

				/*	}*/

				// port tx rate 
				// for (u_int32_t i = 0; i < pCnt; i++)
				// {	
				// 	array[i] = double(m_lastPktSize[i]/ double(dt)*1000000000/(dev->GetDataRate().GetBitRate() / 8));
				// 	if ( *(ptr + i) != 0)
				// {
				// 	printf("port %drate is ",i);
				// 	printf("%0.0f \n", double(*(ptr + i)/ double(dt)*1000000000/(dev->GetDataRate().GetBitRate() / 8)));
				// }}
				// rate_value = double(*(ptr + 7)/ double(dt)*1000000000/(dev->GetDataRate().GetBitRate() / 8));
				
				//get port 7 rate
				// tx_rate = double(*(ptr + 7)/ double(dt)*1000000000/(dev->GetDataRate().GetBitRate() / 8));
				
				//tx_rate = int(*(ptr + 7)/ double(dt)*1000000000);
				// std::cout<<"port rate \t"<<rate_value<<'\n';

				// ingress queue len 
				// for (u_int32_t i = 0; i < pCnt; i++) {
        		// 	for (u_int32_t j = 0; j < qCnt; j++) {
				// 		if(*(*(ptr1 + i) + j) != 0){
				// 			printf("ingress bytes array[%d][%d]",i,j);
				// 			printf("%d \n", *(*(ptr1 + i) + j)); 
				// 		}else{
						
				// 		}	 
        		// 	}
				// }
				//egress qlen 
				// for (u_int32_t i = 0; i < pCnt; i++) {
        		// 	for (u_int32_t j = 0; j < qCnt; j++) {
				// 		if(array2d[i][j] != 0){
				// 			printf("egress bytes array[%d][%d]",i,j);
				// 			printf("%d \n", array2d[i][j]); 
				// 		}else{
						
				// 		}	 
        		// 	}
				// }
				// std::cout<<"qlen \t"<<qlen_value<<"\n";
				
				//get ecn threshold (every port configure the same threshold)


				// std::cout<<"current ecn \t"<<min_thresh<<"\t"<<max_thresh<<"\t"<<pmax_thresh<<"\n";

				// m_mmu->ConfigEcn()
				// std::cout<<" ecn marking rate"<<(m_c/double(1e11))<<"bps \n";
				
				
				
				//tx_rate_m = int(m_c);
				// std::cout<<" ecn marking rate \t"<<tx_rate_m<<"\n";
				// printf("\n");



			}
		}
	}
	m_lastPktSize[ifIndex] = p->GetSize();
	m_lastPktTs[ifIndex] = Simulator::Now().GetTimeStep();


}

int SwitchNode::logres_shift(int b, int l){
	static int data[] = {0,0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5};
	return l - data[b];
}

int SwitchNode::log2apprx(int x, int b, int m, int l){
	int x0 = x;
	int msb = int(log2(x)) + 1;
	if (msb > m){
		x = (x >> (msb - m) << (msb - m));
		#if 0
		x += + (1 << (msb - m - 1));
		#else
		int mask = (1 << (msb-m)) - 1;
		if ((x0 & mask) > (rand() & mask))
			x += 1<<(msb-m);
		#endif
	}
	return int(log2(x) * (1<<logres_shift(b, l)));
}

} /* namespace ns3 */
