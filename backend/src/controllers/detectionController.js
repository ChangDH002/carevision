const { PrismaClient } = require('@prisma/client');
const { sendAlert } = require('../services/fcm');
const prisma = new PrismaClient();

const getLogs = async (req, res) => {
  const { patientId, type } = req.query;
  const logs = await prisma.detectionLog.findMany({
    where: {
      ...(patientId && { patientId: Number(patientId) }),
      ...(type && { type }),
      patient: { userId: req.userId },
    },
    include: { patient: { select: { name: true } } },
    orderBy: { detectedAt: 'desc' },
    take: 100,
  });
  res.json(logs);
};

const createLog = async (req, res) => {
  const { patientId, type, confidence } = req.body;
  if (!patientId || !type || confidence === undefined)
    return res.status(400).json({ message: '필수 항목이 누락되었습니다.' });

  const log = await prisma.detectionLog.create({
    data: { patientId, type, confidence },
    include: { patient: { include: { user: true } } },
  });

  // 낙상 감지 시 FCM 알림 전송
  // AI 서버가 'FALL' / 'fall' 어떤 케이스로 보내도 매칭되도록 정규화해서 비교한다.
  // (이전: type === 'fall' 만 비교 → AI가 'FALL'을 보내면 알림이 누락되는 문제)
  if (typeof type === 'string' && type.toLowerCase() === 'fall') {
    const fcmToken = log.patient.user.fcmToken;
    if (fcmToken) {
      await sendAlert(fcmToken, {
        title: '낙상 감지!',
        body: `${log.patient.name} 님의 낙상이 감지되었습니다.`,
      });
    }
  }

  res.status(201).json(log);
};

module.exports = { getLogs, createLog };
