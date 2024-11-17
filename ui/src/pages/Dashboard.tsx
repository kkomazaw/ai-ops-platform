import React, { useState, useEffect } from 'react';
import {
  DashboardLayout,
  SystemOverview,
  ActiveIncidents,
  ServiceHealth,
  MetricsPanel,
  AlertsPanel,
} from '../components/dashboard';
import { useSystemMetrics, useIncidents, useAlerts } from '../hooks';
import { Card, Grid, Container } from '@/components/ui';

export const Dashboard: React.FC = () => {
  const { metrics, loading: metricsLoading } = useSystemMetrics();
  const { incidents, loading: incidentsLoading } = useIncidents();
  const { alerts, loading: alertsLoading } = useAlerts();

  return (
    <DashboardLayout>
      <Container>
        <Grid container spacing={3}>
          {/* システム概要 */}
          <Grid item xs={12}>
            <SystemOverview
              loading={metricsLoading}
              metrics={metrics}
            />
          </Grid>

          {/* アクティブなインシデント */}
          <Grid item xs={12} md={6}>
            <ActiveIncidents
              loading={incidentsLoading}
              incidents={incidents}
            />
          </Grid>

          {/* サービスの健全性 */}
          <Grid item xs={12} md={6}>
            <ServiceHealth
              loading={metricsLoading}
              services={metrics?.services || []}
            />
          </Grid>

          {/* メトリクスパネル */}
          <Grid item xs={12} md={8}>
            <MetricsPanel
              loading={metricsLoading}
              metrics={metrics}
            />
          </Grid>

          {/* アラートパネル */}
          <Grid item xs={12} md={4}>
            <AlertsPanel
              loading={alertsLoading}
              alerts={alerts}
            />
          </Grid>
        </Grid>
      </Container>
    </DashboardLayout>
  );
};