import React, { useState } from 'react';
import {
  IncidentList,
  IncidentDetails,
  IncidentFilters,
  IncidentTimeline,
} from '../components/incidents';
import { useIncidents } from '../hooks';
import { Card, Grid, Button, Dialog } from '@/components/ui';

export const Incidents: React.FC = () => {
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(null);
  const [filterParams, setFilterParams] = useState({
    status: 'all',
    severity: 'all',
    timeRange: '24h'
  });

  const { incidents, loading, error } = useIncidents(filterParams);
  const selectedIncident = incidents?.find(i => i.id === selectedIncidentId);

  return (
    <div className="p-6">
      <div className="mb-6 flex justify-between items-center">
        <h1 className="text-2xl font-semibold text-gray-900">Incidents</h1>
        <Button
          variant="contained"
          color="primary"
          onClick={() => {/* Handle create */}}
        >
          Create Manual Incident
        </Button>
      </div>

      <Grid container spacing={3}>
        <Grid item xs={12} lg={4}>
          <Card>
            <IncidentFilters
              filters={filterParams}
              onChange={setFilterParams}
            />
          </Card>
        </Grid>

        <Grid item xs={12} lg={8}>
          <Card>
            <IncidentList
              incidents={incidents}
              loading={loading}
              error={error}
              onSelectIncident={setSelectedIncidentId}
              selectedIncidentId={selectedIncidentId}
            />
          </Card>
        </Grid>
      </Grid>

      {selectedIncident && (
        <Dialog
          open={!!selectedIncidentId}
          onClose={() => setSelectedIncidentId(null)}
          maxWidth="lg"
          fullWidth
        >
          <div className="p-6">
            <IncidentDetails
              incident={selectedIncident}
              onClose={() => setSelectedIncidentId(null)}
            />
            <IncidentTimeline
              incidentId={selectedIncident.id}
            />
          </div>
        </Dialog>
      )}
    </div>
  );
};

// インシデントリストコンポーネント
const IncidentList: React.FC<{
  incidents: Incident[];
  loading: boolean;
  error: any;
  onSelectIncident: (id: string) => void;
  selectedIncidentId: string | null;
}> = ({
  incidents,
  loading,
  error,
  onSelectIncident,
  selectedIncidentId
}) => {
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error loading incidents</div>;

  return (
    <div className="divide-y divide-gray-200">
      {incidents.map((incident) => (
        <div
          key={incident.id}
          className={`p-4 cursor-pointer hover:bg-gray-50 ${
            selectedIncidentId === incident.id ? 'bg-blue-50' : ''
          }`}
          onClick={() => onSelectIncident(incident.id)}
        >
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-lg font-medium text-gray-900">
                {incident.title}
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                {incident.description}
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <SeverityBadge severity={incident.severity} />
              <StatusBadge status={incident.status} />
            </div>
          </div>
          <div className="mt-2 text-sm text-gray-500">
            Created {new Date(incident.created_at).toLocaleString()}
          </div>
        </div>
      ))}
    </div>
  );
};

// インシデント詳細コンポーネント
const IncidentDetails: React.FC<{
  incident: Incident;
  onClose: () => void;
}> = ({ incident, onClose }) => {
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-start">
        <h2 className="text-xl font-semibold text-gray-900">
          {incident.title}
        </h2>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-500"
        >
          <XIcon className="h-6 w-6" />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <h3 className="text-sm font-medium text-gray-500">Status</h3>
          <StatusBadge status={incident.status} />
        </div>
        <div>
          <h3 className="text-sm font-medium text-gray-500">Severity</h3>
          <SeverityBadge severity={incident.severity} />
        </div>
      </div>

      <div>
        <h3 className="text-sm font-medium text-gray-500">Description</h3>
        <p className="mt-1 text-sm text-gray-900">{incident.description}</p>
      </div>

      <div>
        <h3 className="text-sm font-medium text-gray-500">Actions</h3>
        <div className="mt-2 flex space-x-2">
          <Button
            variant="outlined"
            color="primary"
            onClick={() => {/* Handle action */}}
          >
            Acknowledge
          </Button>
          <Button
            variant="contained"
            color="success"
            onClick={() => {/* Handle action */}}
          >
            Resolve
          </Button>
          <Button
            variant="outlined"
            color="error"
            onClick={() => {/* Handle action */}}
          >
            Escalate
          </Button>
        </div>
      </div>
    </div>
  );
};